"""Load the official DermaMNIST splits as NumPy arrays and PyTorch data loaders."""

import numpy as np
import torch
from medmnist import DermaMNIST
from torch.utils.data import DataLoader, TensorDataset

SPLITS = ("train", "val", "test")
N_CLASSES = 7


def load_split(split: str, size: int = 28) -> tuple[np.ndarray, np.ndarray]:
    """Return images (N, size, size, 3) uint8 and labels (N,) int64 for one split."""
    if split not in SPLITS:
        raise ValueError(f"Invalid split {split!r}. Expected one of {SPLITS}.")
    dataset = DermaMNIST(split=split, size=size, download=True)
    images = dataset.imgs
    labels = dataset.labels.reshape(-1).astype(np.int64)  # (N, 1) -> (N,)
    return images, labels


def load_splits(size: int = 28) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {"train": (X, y), "val": (X, y), "test": (X, y)}."""
    return {split: load_split(split, size=size) for split in SPLITS}


def to_tensors(images: np.ndarray, labels: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """(N, H, W, 3) uint8 -> (N, 3, H, W) float32 in [0, 1]; labels -> int64 tensor."""
    x = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0  # channels first
    y = torch.from_numpy(labels)
    return x, y


def make_loaders(batch_size: int = 64, size: int = 28, seed: int = 0) -> dict[str, DataLoader]:
    """One DataLoader per split; only the training loader shuffles, in a seeded order."""
    loaders = {}
    for split, (images, labels) in load_splits(size).items():
        dataset = TensorDataset(*to_tensors(images, labels))
        shuffle = split == "train"
        generator = torch.Generator().manual_seed(seed) if shuffle else None
        loaders[split] = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, generator=generator
        )
    return loaders


def class_weights(labels: np.ndarray, n_classes: int = N_CLASSES) -> torch.Tensor:
    """Inverse-frequency weights: the same formula as scikit-learn's class_weight="balanced"."""
    counts = np.bincount(labels, minlength=n_classes)
    return torch.tensor(counts.sum() / (n_classes * counts), dtype=torch.float32)
