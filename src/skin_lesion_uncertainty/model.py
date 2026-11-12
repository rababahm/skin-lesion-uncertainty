"""Convolutional networks for 28 x 28 DermaMNIST images."""

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class SmallCNN(nn.Module):
    """Two convolution blocks, then a linear layer to the class logits.

    Shapes for a 28 x 28 input:
        [N, 3, 28, 28] -> block 1 -> [N, 32, 14, 14] -> block 2 -> [N, 64, 7, 7]
        -> flatten -> [N, 3136] -> dropout -> linear -> [N, n_classes]
    """

    def __init__(self, n_classes: int = 7, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 32 filters of 3x3; padding keeps 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28 -> 14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14 -> 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),  # active in train() mode only
            nn.Linear(64 * 7 * 7, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))  # logits, no softmax


class ImageNetNormalize(nn.Module):
    """Rescale [0, 1] images to the statistics the pretrained weights were trained on."""

    def __init__(self) -> None:
        super().__init__()
        # Buffers move with .to(device) and are saved in the state_dict, but are not trained.
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


def pretrained_resnet18(n_classes: int = 7, pretrained: bool = True) -> nn.Module:
    """ResNet-18 with ImageNet weights and a new final layer for n_classes; all layers train."""
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, n_classes)  # replace the 1000-class ImageNet head
    return nn.Sequential(ImageNetNormalize(), model)
