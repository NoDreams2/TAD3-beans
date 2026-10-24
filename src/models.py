"""Три модели разной сложности."""
import torch.nn as nn
from torchvision import models


class SimpleCNN(nn.Module):
    """Своя свёрточная сеть: 4 блока «свёртка → нормализация → ReLU → пулинг»
    и линейный слой. Около 0,24 млн параметров, обучается с нуля."""

    def __init__(self, num_classes: int):
        super().__init__()
        layers, ch_in = [], 3
        for ch_out in (32, 64, 128, 128):
            layers += [nn.Conv2d(ch_in, ch_out, 3, padding=1), nn.BatchNorm2d(ch_out),
                       nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            ch_in = ch_out
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(name: str, num_classes: int, pretrained: bool) -> nn.Module:
    if name == "simplecnn":
        return SimpleCNN(num_classes)
    if name == "resnet18":
        m = models.resnet18(weights="DEFAULT" if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if name == "mobilenet_v3":
        m = models.mobilenet_v3_small(weights="DEFAULT" if pretrained else None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
        return m
    raise ValueError(f"Неизвестная модель: {name}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
