from __future__ import annotations

import io
import os
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta, timezone
from ipaddress import ip_address
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from torch.utils.data import TensorDataset
from werkzeug.serving import make_server

from fl_platform.dashboard.app import create_app as create_dashboard_app
from fl_platform.deploy.fl_client import create_fl_client_app
from fl_platform.deploy.fl_server import create_fl_server_app
from fl_platform.tracking import get_rounds


API_TOKEN = "integration-test-token-0123456789"


class FederatedRoundIntegrationTest(unittest.TestCase):
    def test_two_clients_upload_custom_npz_and_complete_round(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            db_path = root / "tracking.db"
            server_app, _ = create_fl_server_app({
                "model": "simple_cnn", "num_classes": 2, "dataset": "cifar10",
                "rounds": 2, "min_clients": 2, "clients_per_round": 2,
                "strategy": "fedavg", "db_path": str(db_path),
                "task_id": "integration", "api_token": API_TOKEN,
            })
            unauthenticated = server_app.test_client().get("/status")
            self.assertEqual(unauthenticated.status_code, 401)

            cert_path, key_path = _create_test_certificate(root)
            http_server = make_server("127.0.0.1", 0, server_app, threaded=True,
                                      ssl_context=(str(cert_path), str(key_path)))
            server_thread = threading.Thread(target=http_server.serve_forever, daemon=True)
            server_thread.start()
            server_url = f"https://127.0.0.1:{http_server.server_port}"
            try:
                with patch.dict(os.environ, {"FL_TLS_CA": str(cert_path)}):
                    self._run_two_custom_clients(root, server_url, db_path)
            finally:
                http_server.shutdown()
                server_thread.join(timeout=5)

    def _run_two_custom_clients(self, root: Path, server_url: str, db_path: Path) -> None:
        nodes = []
        for client_id, partition_id in (("client_0", 0), ("client_1", 1)):
            client_conf = {
                "client_id": client_id, "server_url": server_url,
                "my_address": "localhost", "model": "simple_cnn",
                "num_classes": 2, "dataset": "cifar10", "partition_id": partition_id,
                "num_clients_total": 2, "split_type": "iid", "batch_size": 2,
                "local_epoch": 1, "lr": 0.01, "optimizer": "sgd", "gpu": 0,
                "seed": 42, "alpha": 0.5, "api_token": API_TOKEN,
                "upload_dir": str(root / "uploads"),
            }
            client_app, node = create_fl_client_app(client_conf)
            nodes.append(node)
            image_data = np.random.default_rng(partition_id).integers(
                0, 256, size=(4, 32, 32, 3), dtype=np.uint8
            )
            labels = np.array([0, 1, 0, 1], dtype=np.int64)
            archive = io.BytesIO()
            np.savez(archive, x_train=image_data, y_train=labels)
            upload = client_app.test_client().post(
                "/upload-dataset",
                headers={"Authorization": f"Bearer {API_TOKEN}"},
                data={"file": (io.BytesIO(archive.getvalue()), "client-data.npz")},
                content_type="multipart/form-data",
            )
            self.assertEqual(upload.status_code, 200, upload.get_data(as_text=True))
            self.assertEqual(upload.get_json()["dataset"], "custom")
            self.assertEqual(len(node._get_train_loader().dataset), 2)
            node.register()

        nodes[0].train_round(1)
        first_status = nodes[0].http.get(f"{server_url}/status").json()
        self.assertEqual(first_status["round"], 1)
        self.assertEqual(first_status["updates_received"], 1)

        nodes[1].train_round(1)
        final_status = nodes[1].http.get(f"{server_url}/status").json()
        self.assertEqual(final_status["completed_rounds"], 1)
        self.assertEqual(final_status["round"], 2)
        rounds = get_rounds(db_path, "integration")
        self.assertEqual(len(rounds), 1)
        self.assertEqual(rounds[0]["clients"], 2)

    def test_dashboard_has_simulation_form_and_requires_basic_auth_when_configured(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = create_dashboard_app(
                db_path=str(Path(temporary) / "dashboard.db"),
                dashboard_user="operator", dashboard_password="long-test-password",
            )
            self.assertEqual(app.test_client().get("/").status_code, 401)
            response = app.test_client().get("/", headers={
                "Authorization": "Basic b3BlcmF0b3I6bG9uZy10ZXN0LXBhc3N3b3Jk"
            })
            self.assertEqual(response.status_code, 200)
            self.assertIn(b"Run local simulation", response.data)
            self.assertIn(b"/api/simulation/start", response.data)

    @patch("fl_platform.coordinator._targets", return_value=[0, 1, 0, 1])
    @patch("fl_platform.coordinator._load_dataset")
    def test_dashboard_simulation_runs_and_records_a_round(self, load_dataset, _targets):
        with tempfile.TemporaryDirectory() as temporary:
            images = torch.rand(4, 3, 32, 32)
            labels = torch.tensor([0, 1, 0, 1])
            load_dataset.return_value = TensorDataset(images, labels)
            db_path = Path(temporary) / "simulation.db"
            app = create_dashboard_app(db_path=str(db_path))
            client = app.test_client()
            response = client.post("/api/simulation/start", json={
                "dataset": "cifar10", "model": "simple_cnn", "num_classes": 2,
                "rounds": 1, "num_clients": 2, "local_epoch": 1,
                "batch_size": 2, "lr": 0.01, "strategy": "fedavg",
            })
            self.assertEqual(response.status_code, 202, response.get_data(as_text=True))
            started = response.get_json()
            deadline = time.time() + 15
            state = client.get("/api/simulation/status").get_json()
            while state["status"] == "running" and time.time() < deadline:
                time.sleep(0.05)
                state = client.get("/api/simulation/status").get_json()
            self.assertEqual(state["status"], "completed", state)
            rounds = get_rounds(db_path, started["task_id"])
            self.assertEqual(len(rounds), 1)
            self.assertEqual(rounds[0]["clients"], 2)


def _create_test_certificate(directory: Path) -> tuple[Path, Path]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder().subject_name(name).issuer_name(name)
        .public_key(key.public_key()).serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(x509.SubjectAlternativeName([
            x509.DNSName("localhost"), x509.IPAddress(ip_address("127.0.0.1"))
        ]), critical=False)
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .sign(key, hashes.SHA256())
    )
    cert_path, key_path = directory / "test.crt", directory / "test.key"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM,
                                           serialization.PrivateFormat.PKCS8,
                                           serialization.NoEncryption()))
    return cert_path, key_path


if __name__ == "__main__":
    unittest.main()
