"""Small in-process simulation runner used by the dashboard simulation form."""

from __future__ import annotations

import copy
import threading
import uuid
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from fl_platform.deploy.fl_client import _load_dataset, _partition_indices, _targets
from fl_platform.models import create_model
from fl_platform.tracking import record_round, record_run


class SimulationManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.lock = threading.RLock()
        self.state: dict[str, Any] = {"status": "idle", "task_id": None, "round": 0,
                                      "total_rounds": 0, "error": None}
        self.thread: threading.Thread | None = None

    def start(self, values: dict[str, Any]) -> dict[str, Any]:
        config = _validate_simulation_config(values)
        with self.lock:
            if self.state["status"] == "running":
                raise RuntimeError("A simulation is already running")
            task_id = f"simulation-{uuid.uuid4().hex[:10]}"
            config["task_id"] = task_id
            self.state = {"status": "running", "task_id": task_id, "round": 0,
                          "total_rounds": config["rounds"], "error": None,
                          "dataset": config["dataset"], "model": config["model"]}
            self.thread = threading.Thread(target=self._run, args=(config,), daemon=True)
            self.thread.start()
            return dict(self.state)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.state)

    def _run(self, config: dict[str, Any]) -> None:
        try:
            torch.manual_seed(config["seed"])
            device = torch.device("cuda" if config["gpu"] > 0 and torch.cuda.is_available() else "cpu")
            model = create_model(config["model"], config["num_classes"], config["dataset"]).to(device)
            dataset = _load_dataset(config["dataset"], config["data_dir"])
            labels = _targets(dataset)
            record_run(self.db_path, config["task_id"], config["model"],
                       config["dataset"], config["strategy"])
            partitions = [
                _partition_indices(labels, client_id, config["num_clients"],
                                   config["split_type"], config["seed"], config["alpha"])
                for client_id in range(config["num_clients"])
            ]
            self._set_state(status="running", message="Dataset ready")
            for round_number in range(1, config["rounds"] + 1):
                updates = []
                for indices in partitions:
                    if not indices:
                        continue
                    local_model = copy.deepcopy(model).to(device)
                    original = {name: tensor.detach().clone()
                                for name, tensor in local_model.named_parameters()}
                    loader = DataLoader(Subset(dataset, indices), batch_size=config["batch_size"],
                                        shuffle=True, num_workers=0)
                    loss, accuracy = _local_train(local_model, loader, config, original, device)
                    updates.append((local_model.state_dict(), len(indices), loss, accuracy))
                if not updates:
                    raise RuntimeError("No simulation clients received training samples")
                model.load_state_dict(_aggregate(updates, config["strategy"]))
                total_samples = sum(update[1] for update in updates)
                average_loss = sum(update[2] * update[1] for update in updates) / total_samples
                average_accuracy = sum(update[3] * update[1] for update in updates) / total_samples
                record_round(self.db_path, config["task_id"], round_number, len(updates),
                             average_loss, average_accuracy)
                self._set_state(round=round_number, loss=average_loss, accuracy=average_accuracy,
                                clients=len(updates), message=f"Round {round_number} complete")
            self._set_state(status="completed", message="Simulation complete")
        except Exception as exc:
            self._set_state(status="failed", error=str(exc), message="Simulation failed")

    def _set_state(self, **values: Any) -> None:
        with self.lock:
            self.state.update(values)


def _validate_simulation_config(values: dict[str, Any]) -> dict[str, Any]:
    allowed_models = {"simple_cnn", "fedavg_cnn", "fedbn_cnn", "floco_net", "lenet",
                      "digits5cnn", "mlp", "logistic", "rnn", "gru",
                      "resnet18", "resnet34", "resnet50", "resnet101"}
    allowed_datasets = {"cifar10", "cifar100", "mnist", "fashionmnist", "svhn"}
    allowed_splits = {"iid", "dirichlet", "class_based"}
    allowed_strategies = {"fedavg", "fedprox", "fedmedian"}
    result = {
        "task_id": "",
        "dataset": str(values.get("dataset", "cifar10")).lower(),
        "model": str(values.get("model", "simple_cnn")).lower(),
        "num_classes": int(values.get("num_classes", 100 if str(values.get("dataset", "")).lower() == "cifar100" else 10)),
        "rounds": int(values.get("rounds", 3)),
        "num_clients": int(values.get("num_clients", 2)),
        "local_epoch": int(values.get("local_epoch", 1)),
        "batch_size": int(values.get("batch_size", 32)),
        "lr": float(values.get("lr", 0.01)),
        "optimizer": str(values.get("optimizer", "sgd")).lower(),
        "strategy": str(values.get("strategy", "fedavg")).lower(),
        "split_type": str(values.get("split_type", "iid")).lower(),
        "alpha": float(values.get("alpha", 0.5)),
        "seed": int(values.get("seed", 42)),
        "gpu": int(values.get("gpu", 0)),
        "data_dir": str(values.get("data_dir", "./data")),
    }
    if result["dataset"] not in allowed_datasets or result["model"] not in allowed_models:
        raise ValueError("Unsupported simulation model or dataset")
    if result["split_type"] not in allowed_splits or result["strategy"] not in allowed_strategies:
        raise ValueError("Unsupported simulation split type or aggregation strategy")
    if result["optimizer"] not in {"sgd", "adam"}:
        raise ValueError("optimizer must be sgd or adam")
    if not 1 <= result["rounds"] <= 100 or not 1 <= result["num_clients"] <= 32:
        raise ValueError("rounds must be 1..100 and num_clients must be 1..32")
    if not 1 <= result["local_epoch"] <= 20 or not 1 <= result["batch_size"] <= 1024:
        raise ValueError("local_epoch must be 1..20 and batch_size must be 1..1024")
    if result["lr"] <= 0 or result["alpha"] <= 0 or result["num_classes"] < 2:
        raise ValueError("lr, alpha, and num_classes must be positive (num_classes >= 2)")
    return result


def _local_train(model: nn.Module, loader: DataLoader, config: dict[str, Any],
                 original: dict[str, torch.Tensor], device: torch.device) -> tuple[float, float]:
    optimizer = (torch.optim.Adam(model.parameters(), lr=config["lr"])
                 if config["optimizer"] == "adam"
                 else torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9))
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = seen = 0
    model.train()
    for _ in range(config["local_epoch"]):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device).long()
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if config["strategy"] == "fedprox":
                penalty = sum((parameter - original[name]).pow(2).sum()
                              for name, parameter in model.named_parameters())
                loss = loss + 0.005 * penalty / 2
            loss.backward()
            optimizer.step()
            count = labels.size(0)
            total_loss += float(loss.detach()) * count
            correct += int((logits.argmax(1) == labels).sum())
            seen += count
    return total_loss / max(seen, 1), correct / max(seen, 1)


def _aggregate(updates: list[tuple[dict[str, torch.Tensor], int, float, float]],
               strategy: str) -> dict[str, torch.Tensor]:
    total = sum(update[1] for update in updates)
    result = {}
    for name in updates[0][0]:
        values = [update[0][name] for update in updates]
        if not values[0].is_floating_point():
            result[name] = torch.stack(values).max(dim=0).values
        elif strategy == "fedmedian":
            result[name] = torch.stack(values).median(dim=0).values
        else:
            result[name] = sum((value * (count / total)
                                for value, (_, count, _, _) in zip(values, updates)),
                               torch.zeros_like(values[0]))
    return result
