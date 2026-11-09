"""Model outputs for evaluation, as NumPy arrays."""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def collect_logits(
    model: nn.Module, loader: DataLoader, device: str
) -> tuple[np.ndarray, np.ndarray]:
    """Logits (n, n_classes) and labels (n,) for every image, in loader order."""
    model.eval()
    logits_list = []
    labels_list = []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)

        logits_list.append(logits.cpu())
        labels_list.append(labels.cpu())

    return (
        torch.cat(logits_list, dim=0).numpy(),
        torch.cat(labels_list, dim=0).numpy(),
    )
