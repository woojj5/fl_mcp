"""JSON-safe tensor serialization for demo deployments."""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Mapping

import numpy as np
import torch


def encode_weights(state: Mapping[str, torch.Tensor]) -> str:
    buffer = BytesIO()
    arrays = {name: tensor.detach().cpu().numpy() for name, tensor in state.items()}
    np.savez_compressed(buffer, **arrays)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_weights(payload: str, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    try:
        raw = base64.b64decode(payload, validate=True)
        with np.load(BytesIO(raw), allow_pickle=False) as archive:
            return {name: torch.from_numpy(archive[name].copy()).to(device) for name in archive.files}
    except Exception as exc:
        raise ValueError("Invalid encoded model weights") from exc
