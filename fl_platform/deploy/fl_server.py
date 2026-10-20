"""Flask coordinator for a small, single-process federated learning demo."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
import torch
from flask import Flask, jsonify, request

from fl_platform.deploy.protocol import decode_weights, encode_weights
from fl_platform.models import create_model
from fl_platform.security import install_api_token_auth
from fl_platform.tracking import record_client, record_round, record_run


class Coordinator:
    def __init__(self, conf: dict[str, Any]):
        self.conf = conf
        self.model = create_model(conf["model"], int(conf["num_classes"]), conf["dataset"])
        self.clients: dict[str, dict[str, Any]] = {}
        self.updates: dict[str, dict[str, Any]] = {}
        self.round = 1
        self.completed_rounds = 0
        self.lock = threading.RLock()
        self.server_momentum: dict[str, torch.Tensor] = {}
        self.server_second_moment: dict[str, torch.Tensor] = {}
        self.status = "waiting_for_clients"
        self.last_error: str | None = None
        record_run(conf["db_path"], conf["task_id"], conf["model"],
                   conf["dataset"], conf["strategy"])

    def status_payload(self) -> dict[str, Any]:
        with self.lock:
            return {
                "status": self.status,
                "task_id": self.conf["task_id"],
                "round": self.round,
                "completed_rounds": self.completed_rounds,
                "total_rounds": self.conf["rounds"],
                "registered_clients": len(self.clients),
                "min_clients": self.conf["min_clients"],
                "clients_per_round": self.conf["clients_per_round"],
                "updates_received": len(self.updates),
                "strategy": self.conf["strategy"],
                "last_error": self.last_error,
            }

    def register(self, client_id: str, address: str) -> None:
        with self.lock:
            self.clients[client_id] = {"client_id": client_id, "address": address}
            if len(self.clients) >= self.conf["min_clients"] and self.status != "completed":
                self.status = "training"

    def submit(self, payload: dict[str, Any]) -> tuple[bool, str]:
        client_id = str(payload.get("client_id", ""))
        if not client_id or client_id not in self.clients:
            return False, "client must register before uploading weights"
        if payload.get("round") != self.round:
            return False, f"stale round; server expects round {self.round}"
        try:
            weights = decode_weights(payload["weights"])
            current = self.model.state_dict()
            if weights.keys() != current.keys() or any(
                weights[key].shape != current[key].shape for key in current
            ):
                return False, "uploaded model shape does not match server model"
        except (KeyError, TypeError, ValueError) as exc:
            return False, str(exc)

        with self.lock:
            if client_id in self.updates:
                return False, "client already submitted for this round"
            self.updates[client_id] = {
                "weights": weights,
                "samples": max(1, int(payload.get("samples", 1))),
                "loss": _optional_float(payload.get("loss")),
                "accuracy": _optional_float(payload.get("accuracy")),
            }
            metrics = self.updates[client_id]
            record_client(self.conf["db_path"], self.conf["task_id"], client_id,
                          self.round, metrics["samples"], metrics["loss"], metrics["accuracy"])
            threshold = max(self.conf["min_clients"],
                            min(self.conf["clients_per_round"], len(self.clients)))
            if len(self.updates) >= threshold:
                self._aggregate()
        return True, "weights accepted"

    def _aggregate(self) -> None:
        updates = list(self.updates.values())
        if not updates:
            return
        strategy = self.conf["strategy"].lower()
        old = self.model.state_dict()
        sample_counts = torch.tensor([item["samples"] for item in updates], dtype=torch.float64)
        sample_weights = sample_counts / sample_counts.sum()
        krum_winner = None
        if strategy == "krum" and len(updates) > 1:
            scores = []
            for candidate_index, candidate in enumerate(updates):
                score = 0.0
                for peer_index, peer in enumerate(updates):
                    if candidate_index != peer_index:
                        score += sum((candidate["weights"][name].float() - peer["weights"][name].float())
                                     .pow(2).sum().item() for name in old)
                scores.append(score)
            krum_winner = int(np.argmin(scores))
        aggregated: dict[str, torch.Tensor] = {}
        for name, old_value in old.items():
            values = [item["weights"][name].to(old_value.dtype) for item in updates]
            if not old_value.is_floating_point():
                aggregated[name] = torch.stack(values).max(dim=0).values
                continue
            if strategy == "fedmedian":
                aggregated[name] = torch.stack(values).median(dim=0).values
            elif strategy == "krum":
                aggregated[name] = values[krum_winner or 0]
            else:
                aggregate = torch.zeros_like(old_value)
                for weight, value in zip(sample_weights, values):
                    aggregate.add_(value, alpha=float(weight))
                if strategy == "fedavgm":
                    delta = aggregate - old_value
                    velocity = self.server_momentum.get(name, torch.zeros_like(delta))
                    velocity = 0.9 * velocity + delta
                    self.server_momentum[name] = velocity
                    aggregate = old_value + velocity
                elif strategy in {"fedadam", "fedyogi"}:
                    delta = aggregate - old_value
                    first = self.server_momentum.get(name, torch.zeros_like(delta))
                    first = 0.9 * first + 0.1 * delta
                    self.server_momentum[name] = first
                    second = self.server_second_moment.get(name, torch.zeros_like(delta))
                    squared_delta = delta.pow(2)
                    if strategy == "fedyogi":
                        second = second - 0.01 * squared_delta * torch.sign(second - squared_delta)
                        second = second.clamp_min(0)
                    else:
                        second = 0.99 * second + 0.01 * squared_delta
                    self.server_second_moment[name] = second
                    step = float(self.conf.get("eta", 0.01)) * first / (second.sqrt() + 1e-8)
                    aggregate = old_value + step
                elif strategy == "dp_fedavg":
                    delta = aggregate - old_value
                    norm = delta.float().norm().clamp_min(1e-12)
                    delta = delta * min(1.0, float(self.conf.get("clipping_norm", 1.0)) / norm.item())
                    noise = torch.randn_like(delta) * float(self.conf.get("noise_multiplier", 1.0)) * 1e-3
                    aggregate = old_value + delta + noise
                aggregated[name] = aggregate
        self.model.load_state_dict(aggregated)
        losses = [item["loss"] for item in updates if item["loss"] is not None]
        accuracies = [item["accuracy"] for item in updates if item["accuracy"] is not None]
        record_round(self.conf["db_path"], self.conf["task_id"], self.round, len(updates),
                     float(np.mean(losses)) if losses else None,
                     float(np.mean(accuracies)) if accuracies else None)
        self.completed_rounds += 1
        self.round += 1
        self.updates.clear()
        self.status = "completed" if self.completed_rounds >= self.conf["rounds"] else "training"

    def advance(self) -> bool:
        with self.lock:
            if not self.updates:
                return False
            self._aggregate()
            return True

    def reset(self) -> None:
        with self.lock:
            self.model = create_model(self.conf["model"], int(self.conf["num_classes"]), self.conf["dataset"])
            self.clients.clear()
            self.updates.clear()
            self.round = 1
            self.completed_rounds = 0
            self.status = "waiting_for_clients"
            self.last_error = None


def create_fl_server_app(conf: dict[str, Any]):
    app = Flask("fl_server")
    app.config["MAX_CONTENT_LENGTH"] = 256 * 1024 * 1024
    install_api_token_auth(app, conf.get("api_token"))
    coordinator = Coordinator(conf)

    @app.get("/health")
    def health():
        return jsonify({"ok": True, "service": "fl-server"})

    @app.get("/status")
    def status():
        return jsonify(coordinator.status_payload())

    @app.post("/register")
    def register():
        body = request.get_json(silent=True) or {}
        client_id = str(body.get("client_id", "")).strip()
        if not client_id:
            return jsonify({"error": "client_id is required"}), 400
        coordinator.register(client_id, str(body.get("address", "")))
        return jsonify({"ok": True, "client_id": client_id, "status": coordinator.status_payload()})

    @app.get("/global-model")
    def global_model():
        state = coordinator.status_payload()
        if state["registered_clients"] < state["min_clients"]:
            return jsonify({"error": "waiting for minimum clients", **state}), 409
        return jsonify({"round": state["round"], "weights": encode_weights(coordinator.model.state_dict()),
                        "model": conf["model"], "num_classes": conf["num_classes"]})

    @app.post("/upload-weights")
    def upload_weights():
        body = request.get_json(silent=True) or {}
        accepted, message = coordinator.submit(body)
        if not accepted:
            return jsonify({"error": message}), 400
        return jsonify({"ok": True, "message": message, "status": coordinator.status_payload()})

    @app.post("/predict")
    def predict():
        try:
            result = _predict(coordinator.model, request)
        except (ValueError, TypeError, RuntimeError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(result)

    @app.get("/admin/clients")
    def clients():
        with coordinator.lock:
            return jsonify({"clients": list(coordinator.clients.values())})

    @app.post("/admin/next-round")
    def next_round():
        advanced = coordinator.advance()
        if not advanced:
            return jsonify({"error": "no client updates are ready to aggregate"}), 409
        return jsonify({"ok": True, "status": coordinator.status_payload()})

    @app.post("/admin/reset")
    def reset():
        coordinator.reset()
        return jsonify({"ok": True, "status": coordinator.status_payload()})

    return app, coordinator


def _optional_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _predict(model: torch.nn.Module, req) -> dict[str, Any]:
    if "image" in req.files:
        from PIL import Image
        from torchvision import transforms
        image = Image.open(req.files["image"].stream).convert("RGB")
        inputs = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])(image).unsqueeze(0)
    else:
        body = req.get_json(silent=True) or {}
        if "input" not in body:
            raise ValueError("JSON input or multipart image is required")
        inputs = torch.as_tensor(body["input"], dtype=torch.float32)
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(0).unsqueeze(0)
        elif inputs.ndim == 3:
            inputs = inputs.unsqueeze(0) if inputs.shape[0] in (1, 3) else inputs.unsqueeze(1)
        if inputs.ndim != 4:
            raise ValueError("input must have shape [batch, channels, height, width]")
    model.eval()
    try:
        inputs = inputs.to(next(model.parameters()).device)
    except StopIteration:
        pass
    with torch.no_grad():
        logits = model(inputs)
        probabilities = logits.softmax(dim=1)
        top_k = max(1, min(int(req.args.get("top_k", 5)), probabilities.shape[1]))
        values, indices = probabilities.topk(top_k, dim=1)
    return {"predictions": indices.cpu().tolist(), "probabilities": values.cpu().tolist()}
