"""Convolutional networks for 28 x 28 DermaMNIST images."""

import torch
from torch import nn


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
