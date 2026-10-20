"""A Flask client node that trains locally and uploads model updates."""

from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

import numpy as np
import torch
from flask import Flask, jsonify, request
from torch import nn
from torch.utils.data import DataLoader, Subset

from fl_platform.deploy.protocol import decode_weights, encode_weights
from fl_platform.models import create_model
from fl_platform.security import install_api_token_auth, tls_verify_setting


class ClientNode:
    def __init__(self, conf: dict[str, Any]):
        self.conf = conf
        self.device = torch.device("cuda" if conf.get("gpu", 0) > 0 and torch.cuda.is_available() else "cpu")
        self.model = create_model(conf["model"], int(conf["num_classes"]), conf["dataset"]).to(self.device)
        self.running = False
        self.thread: threading.Thread | None = None
        self.lock = threading.RLock()
        self.state: dict[str, Any] = {
            "status": "idle", "client_id": conf["client_id"], "round": 0,
            "loss": None, "accuracy": None, "samples": 0, "last_error": None,
        }
        self._dataset_cache = None
        self._last_submitted_round = 0
        self._custom_dataset_path: Path | None = None
        import requests
        self.http = requests.Session()
        self.http.headers.update({"Authorization": f"Bearer {conf['api_token']}"})
        self.http.verify = tls_verify_setting()

    def register(self) -> None:
        response = self.http.post(
            f"{self.conf['server_url'].rstrip('/')}/register",
            json={"client_id": self.conf["client_id"], "address": self.conf["my_address"]},
            timeout=10,
        )
        response.raise_for_status()

    def status(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.state)

    def set_custom_dataset(self, path: Path) -> None:
        self._custom_dataset_path = path
        self._dataset_cache = None
        with self.lock:
            self.state["dataset"] = "custom"

    def start_loop(self) -> None:
        with self.lock:
            if self.running:
                return
            self.running = True
            self.thread = threading.Thread(target=self._loop, name=f"fl-{self.conf['client_id']}", daemon=True)
            self.thread.start()

    def stop_loop(self) -> None:
        with self.lock:
            self.running = False
            self.state["status"] = "stopping"

    def _loop(self) -> None:
        while self.running:
            try:
                self.register()
                response = self.http.get(f"{self.conf['server_url'].rstrip('/')}/status", timeout=10)
                response.raise_for_status()
                server_state = response.json()
                round_number = int(server_state["round"])
                can_train = (server_state["registered_clients"] >= server_state["min_clients"]
                             and server_state["completed_rounds"] < server_state["total_rounds"])
                if can_train and round_number > self._last_submitted_round:
                    self.train_round(round_number)
                else:
                    time.sleep(2)
            except Exception as exc:
                with self.lock:
                    self.state["status"] = "waiting_for_server"
                    self.state["last_error"] = str(exc)
                time.sleep(3)
        with self.lock:
            self.state["status"] = "stopped"

    def train_round(self, round_number: int | None = None) -> dict[str, Any]:
        with self.lock:
            self.state["status"] = "training"
            self.state["last_error"] = None
        try:
            global_response = self.http.get(f"{self.conf['server_url'].rstrip('/')}/global-model", timeout=30)
            global_response.raise_for_status()
            global_payload = global_response.json()
            current_round = int(global_payload["round"])
            if round_number is not None and current_round != round_number:
                raise RuntimeError(f"server round changed: expected {round_number}, got {current_round}")
            self.model.load_state_dict(decode_weights(global_payload["weights"], self.device))
            proximal_state = {name: parameter.detach().clone()
                              for name, parameter in self.model.named_parameters()}
            server_state = self.http.get(f"{self.conf['server_url'].rstrip('/')}/status", timeout=10)
            server_state.raise_for_status()
            strategy = server_state.json().get("strategy", "fedavg").lower()
            train_loader = self._get_train_loader()
            loss, accuracy = self._train(train_loader, proximal_state, strategy)
            samples = len(train_loader.dataset)
            response = self.http.post(
                f"{self.conf['server_url'].rstrip('/')}/upload-weights",
                json={"client_id": self.conf["client_id"], "round": current_round,
                      "weights": encode_weights(self.model.state_dict()), "samples": samples,
                      "loss": loss, "accuracy": accuracy}, timeout=120,
            )
            response.raise_for_status()
            self._last_submitted_round = current_round
            with self.lock:
                self.state.update(status="waiting", round=current_round, loss=loss,
                                  accuracy=accuracy, samples=samples, last_error=None)
            return self.status()
        except Exception as exc:
            with self.lock:
                self.state.update(status="error", last_error=str(exc))
            raise

    def _get_train_loader(self) -> DataLoader:
        if self._dataset_cache is None:
            if self._custom_dataset_path is not None:
                self._dataset_cache = _load_custom_dataset(
                    self._custom_dataset_path, self.conf["dataset"], int(self.conf["num_classes"])
                )
            else:
                self._dataset_cache = _load_dataset(self.conf["dataset"], self.conf.get("data_dir", "./data"))
        dataset = self._dataset_cache
        labels = _targets(dataset)
        if labels and max(labels) >= int(self.conf["num_classes"]):
            raise RuntimeError("Dataset labels exceed the configured num_classes")
        indices = _partition_indices(labels, int(self.conf["partition_id"]),
                                     int(self.conf["num_clients_total"]),
                                     self.conf["split_type"], int(self.conf["seed"]),
                                     float(self.conf.get("alpha", 0.5)))
        if not indices:
            raise RuntimeError("This client received an empty data partition")
        subset = Subset(dataset, indices)
        return DataLoader(subset, batch_size=int(self.conf["batch_size"]), shuffle=True, num_workers=0)

    def _train(self, loader: DataLoader, proximal_state: dict[str, torch.Tensor],
               strategy: str) -> tuple[float, float]:
        self.model.train()
        if self.conf["optimizer"].lower() == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.conf["lr"]))
        else:
            optimizer = torch.optim.SGD(self.model.parameters(), lr=float(self.conf["lr"]), momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = seen = 0
        for _ in range(max(1, int(self.conf["local_epoch"]))):
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device).long()
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(images)
                loss = criterion(logits, labels)
                if strategy == "fedprox":
                    proximal = sum((parameter - proximal_state[name]).pow(2).sum()
                                   for name, parameter in self.model.named_parameters())
                    loss = loss + float(self.conf.get("proximal_mu", 0.01)) * 0.5 * proximal
                loss.backward()
                optimizer.step()
                batch = labels.size(0)
                total_loss += float(loss.detach()) * batch
                correct += int((logits.argmax(1) == labels).sum())
                seen += batch
        return total_loss / max(1, seen), correct / max(1, seen)


def create_fl_client_app(conf: dict[str, Any]):
    app = Flask(f"fl_client_{conf['client_id']}")
    app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024
    install_api_token_auth(app, conf.get("api_token"))
    node = ClientNode(conf)
    upload_dir = Path(conf.get("upload_dir", "./uploads")) / conf["client_id"]

    @app.get("/health")
    def health():
        return jsonify({"ok": True, "service": "fl-client", "client_id": conf["client_id"]})

    @app.get("/status")
    def status():
        return jsonify(node.status())

    @app.post("/train")
    def train():
        try:
            return jsonify(node.train_round())
        except Exception as exc:
            return jsonify({"error": str(exc), "status": node.status()}), 502

    @app.post("/predict")
    def predict():
        try:
            from fl_platform.deploy.fl_server import _predict
            result = _predict(node.model, request)
            return jsonify(result)
        except (ValueError, TypeError, RuntimeError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/upload-dataset")
    def upload_dataset():
        uploaded = request.files.get("file")
        if uploaded is None or not uploaded.filename:
            return jsonify({"error": "multipart field 'file' is required"}), 400
        suffix = Path(uploaded.filename).suffix.lower()
        if suffix not in {".zip", ".npz", ".csv"}:
            return jsonify({"error": "supported formats: ZIP, NPZ, CSV"}), 400
        upload_dir.mkdir(parents=True, exist_ok=True)
        target = upload_dir / f"dataset-{uuid.uuid4().hex}{suffix}" if suffix == ".zip" else upload_dir / f"dataset{suffix}"
        uploaded.save(target)
        if suffix == ".zip":
            try:
                with ZipFile(target) as archive:
                    root = upload_dir / target.stem
                    members = archive.infolist()
                    if len(members) > 10000 or sum(item.file_size for item in members) > 1024**3:
                        target.unlink(missing_ok=True)
                        return jsonify({"error": "ZIP dataset exceeds extraction limits"}), 413
                    for member in members:
                        destination = (root / member.filename).resolve()
                        if root.resolve() not in destination.parents and destination != root.resolve():
                            target.unlink(missing_ok=True)
                            return jsonify({"error": "unsafe archive path"}), 400
                    archive.extractall(root)
                    node.set_custom_dataset(root)
            except BadZipFile:
                target.unlink(missing_ok=True)
                return jsonify({"error": "invalid ZIP archive"}), 400
        else:
            node.set_custom_dataset(target)
        return jsonify({"ok": True, "filename": target.name,
                        "dataset": "custom", "message": "custom dataset is active for the next local training round"})

    @app.post("/admin/start")
    def start():
        node.start_loop()
        return jsonify({"ok": True, "status": node.status()})

    @app.post("/admin/stop")
    def stop():
        node.stop_loop()
        return jsonify({"ok": True, "status": node.status()})

    return app, node


def _load_dataset(name: str, data_dir: str):
    try:
        from torchvision import datasets, transforms
    except Exception as exc:
        raise RuntimeError("torchvision is required to load built-in datasets") from exc
    key = name.lower()
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    root = str(Path(data_dir))
    choices = {
        "cifar10": (datasets.CIFAR10, {"train": True}),
        "cifar100": (datasets.CIFAR100, {"train": True}),
        "mnist": (datasets.MNIST, {"train": True}),
        "fashionmnist": (datasets.FashionMNIST, {"train": True}),
        "svhn": (datasets.SVHN, {"split": "train"}),
    }
    if key not in choices:
        raise RuntimeError(f"Unsupported built-in dataset: {name}")
    dataset_class, kwargs = choices[key]
    return dataset_class(root=root, download=True, transform=transform, **kwargs)


def _targets(dataset) -> list[int]:
    raw = getattr(dataset, "targets", None)
    if raw is None:
        raw = getattr(dataset, "labels", None)
    if raw is None and hasattr(dataset, "tensors") and len(dataset.tensors) > 1:
        raw = dataset.tensors[1]
    if raw is None:
        raise RuntimeError("Dataset does not expose class labels for partitioning")
    if isinstance(raw, torch.Tensor):
        return raw.cpu().tolist()
    return [int(label) for label in raw]


def _partition_indices(labels: list[int], partition_id: int, num_clients: int,
                       split_type: str, seed: int, alpha: float = 0.5) -> list[int]:
    if num_clients < 1 or not 0 <= partition_id < num_clients:
        raise ValueError("partition-id must be in [0, num-clients-total)")
    generator = torch.Generator().manual_seed(seed)
    all_indices = torch.randperm(len(labels), generator=generator).tolist()
    mode = split_type.lower()
    if mode == "iid":
        return all_indices[partition_id::num_clients]
    if mode == "class_based":
        class_count = max(labels, default=0) + 1
        selected = {partition_id % class_count, (partition_id + 1) % class_count}
        return [idx for idx in all_indices if labels[idx] in selected]
    if mode == "dirichlet":
        if alpha <= 0:
            raise ValueError("Dirichlet alpha must be greater than zero")
        rng = np.random.default_rng(seed)
        per_client: list[list[int]] = [[] for _ in range(num_clients)]
        classes = sorted(set(labels))
        for class_id in classes:
            class_indices = [idx for idx in all_indices if labels[idx] == class_id]
            rng.shuffle(class_indices)
            proportions = rng.dirichlet(np.full(num_clients, alpha))
            boundaries = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
            for client_id, client_indices in enumerate(np.split(class_indices, boundaries)):
                per_client[client_id].extend(client_indices.tolist())
        return per_client[partition_id]
    raise ValueError(f"Unsupported split type: {split_type}")


def _load_custom_dataset(path: Path, dataset_name: str, num_classes: int):
    from torch.utils.data import TensorDataset

    if path.is_dir():
        from torchvision import datasets, transforms
        root = path / "train" if (path / "train").is_dir() else path
        channels = 1 if dataset_name.lower() in {"mnist", "fashionmnist"} else 3
        image_transform = [transforms.Resize((32, 32))]
        image_transform.append(transforms.Grayscale(num_output_channels=1) if channels == 1
                               else transforms.Lambda(lambda image: image.convert("RGB")))
        image_transform.append(transforms.ToTensor())
        try:
            dataset = datasets.ImageFolder(str(root), transform=transforms.Compose(image_transform))
        except (FileNotFoundError, RuntimeError) as exc:
            raise RuntimeError("ZIP dataset must contain class subdirectories of images") from exc
        if len(dataset.classes) > num_classes:
            raise RuntimeError("ImageFolder has more classes than configured num_classes")
        return dataset

    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if not {"x_train", "y_train"}.issubset(archive.files):
                raise RuntimeError("NPZ dataset requires x_train and y_train arrays")
            features = archive["x_train"]
            labels = archive["y_train"]
    elif path.suffix.lower() == ".csv":
        try:
            rows = np.loadtxt(path, delimiter=",")
        except ValueError:
            rows = np.loadtxt(path, delimiter=",", skiprows=1)
        if rows.ndim != 2 or rows.shape[1] < 2:
            raise RuntimeError("CSV requires a label column and at least one feature column")
        labels, features = rows[:, 0], rows[:, 1:]
    else:
        raise RuntimeError(f"Unsupported custom dataset file: {path.suffix}")

    features = _image_tensor(features, dataset_name)
    labels = torch.as_tensor(np.asarray(labels), dtype=torch.long).reshape(-1)
    if len(features) != len(labels) or len(features) == 0:
        raise RuntimeError("Custom dataset features and labels must have equal non-zero rows")
    class_values = torch.unique(labels, sorted=True)
    if len(class_values) > num_classes:
        raise RuntimeError("Custom dataset has more labels than configured num_classes")
    mapping = {int(value): index for index, value in enumerate(class_values.tolist())}
    labels = torch.tensor([mapping[int(value)] for value in labels.tolist()], dtype=torch.long)
    return TensorDataset(features, labels)


def _image_tensor(features: np.ndarray, dataset_name: str) -> torch.Tensor:
    array = np.asarray(features)
    channels = 1 if dataset_name.lower() in {"mnist", "fashionmnist"} else 3
    if array.ndim == 2:
        pixels = array.shape[1]
        side = int(np.sqrt(pixels / channels))
        if side * side * channels != pixels:
            raise RuntimeError("CSV/NPZ features must be flattened square grayscale or RGB images")
        array = array.reshape(len(array), side, side, channels)
    if array.ndim == 3:
        array = array[:, :, :, None]
    if array.ndim != 4:
        raise RuntimeError("Custom image arrays must have 2D images or NHWC/NCHW batches")
    if array.shape[1] in (1, 3) and array.shape[-1] not in (1, 3):
        tensor = torch.as_tensor(array, dtype=torch.float32)
    elif array.shape[-1] in (1, 3):
        tensor = torch.as_tensor(array.transpose(0, 3, 1, 2).copy(), dtype=torch.float32)
    else:
        raise RuntimeError("Custom image arrays must have one or three channels")
    if tensor.max().item() > 1:
        tensor = tensor / 255.0
    if tensor.shape[1] != channels:
        raise RuntimeError(f"Dataset model expects {channels} channel(s), received {tensor.shape[1]}")
    if tensor.shape[2:] != (32, 32):
        tensor = torch.nn.functional.interpolate(tensor, size=(32, 32), mode="bilinear", align_corners=False)
    return tensor
