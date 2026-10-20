from __future__ import annotations

import torch
from torch import nn


class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64 * 4 * 4, 128),
                                        nn.ReLU(), nn.Linear(128, num_classes))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


class LeNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, 5, padding=2), nn.Tanh(), nn.AvgPool2d(2),
            nn.Conv2d(6, 16, 5), nn.Tanh(), nn.AvgPool2d(2), nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(16 * 4 * 4, 120),
                                        nn.Tanh(), nn.Linear(120, 84), nn.Tanh(),
                                        nn.Linear(84, num_classes))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


class MLP(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.network = nn.Sequential(nn.Flatten(), nn.Linear(in_channels * 32 * 32, 256), nn.ReLU(),
                                     nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, num_classes))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def create_model(name: str, num_classes: int = 10, dataset: str = "cifar10") -> nn.Module:
    """Build one of the small models supported by this starter runtime."""
    key = name.lower()
    channels = 1 if dataset.lower() in {"mnist", "fashionmnist"} else 3
    if key in {"simple_cnn", "fedavg_cnn", "fedbn_cnn", "floco_net"}:
        return SimpleCNN(channels, num_classes)
    if key in {"lenet", "digits5cnn"}:
        return LeNet(channels, num_classes)
    if key in {"mlp", "logistic", "rnn", "gru"}:
        return MLP(channels, num_classes)
    if key.startswith("resnet"):
        try:
            from torchvision import models
        except Exception as exc:
            raise RuntimeError("torchvision 모델을 불러올 수 없습니다") from exc
        factory = getattr(models, key, None)
        if factory is None:
            raise ValueError(f"Unsupported model: {name}")
        model = factory(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        if channels == 1:
            model.conv1 = nn.Conv2d(1, model.conv1.out_channels, kernel_size=7,
                                    stride=2, padding=3, bias=False)
        return model
    raise ValueError(f"Unsupported model: {name}")
