"""Load the official DermaMNIST splits as NumPy arrays."""

import numpy as np
from medmnist import DermaMNIST

SPLITS = ("train", "val", "test")


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
