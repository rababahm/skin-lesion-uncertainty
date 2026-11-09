"""Model outputs and calibration metrics for evaluation, as NumPy arrays."""

import numpy as np
import torch
from scipy.special import softmax
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


def top_label(logits: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Confidence (largest probability) and correctness (0 or 1) of each prediction."""
    probs = softmax(logits.astype(np.float64), axis=1)
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(np.int64)
    return confidence, correct


def reliability_bins(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Count, mean confidence and accuracy in equal-width bins on [0, 1]."""
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1.")

    # Bins are [left, right), except the last bin includes 1.
    bin_indices = np.minimum((confidence * n_bins).astype(np.int64), n_bins - 1)

    counts = np.bincount(bin_indices, minlength=n_bins)
    confidence_sums = np.bincount(bin_indices, weights=confidence, minlength=n_bins)
    correct_sums = np.bincount(bin_indices, weights=correct, minlength=n_bins)

    mean_confidence = np.divide(
        confidence_sums,
        counts,
        out=np.zeros(n_bins, dtype=np.float64),
        where=counts > 0,
    )
    accuracy = np.divide(
        correct_sums,
        counts,
        out=np.zeros(n_bins, dtype=np.float64),
        where=counts > 0,
    )

    return counts, mean_confidence, accuracy


def expected_calibration_error(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15
) -> float:
    """Sum over bins of (count / n) * |accuracy - mean confidence|."""
    counts, mean_confidence, accuracy = reliability_bins(confidence, correct, n_bins)
    n = counts.sum()
    if n == 0:
        raise ValueError("No predictions: cannot compute ECE on empty input.")

    nonempty = counts > 0
    return float(
        np.sum((counts[nonempty] / n) * np.abs(accuracy[nonempty] - mean_confidence[nonempty]))
    )
