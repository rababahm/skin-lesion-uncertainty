import math

import numpy as np


def conformal_scores(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """1 - probability of the true class, one score per image."""
    return 1 - probs[np.arange(len(labels)), labels]


def conformal_threshold(scores: np.ndarray, alpha: float) -> float:
    """The ceil((n+1)(1-alpha))-th smallest score; infinity if that rank exceeds n."""
    if not 0 <= alpha < 1:
        raise ValueError("alpha must satisfy 0 <= alpha < 1.")

    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return float("inf")
    return float(np.sort(scores)[k - 1])


def prediction_sets(probs: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean array (n, n_classes): class y is in the set if 1 - p(y) <= threshold."""
    return (1 - probs) <= threshold


def coverage_and_size(sets: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Share of images whose true class is in the set, and the mean set size."""
    n = len(labels)
    coverage = sets[np.arange(n), labels].mean()
    mean_size = sets.sum(axis=1).mean()
    return float(coverage), float(mean_size)


def classwise_thresholds(
    scores: np.ndarray, labels: np.ndarray, n_classes: int, alpha: float
) -> np.ndarray:
    """One threshold per class, from the calibration images of that class (Mondrian)."""
    # A class with too few calibration images gets an infinite threshold: always included.
    return np.array([conformal_threshold(scores[labels == k], alpha) for k in range(n_classes)])


def classwise_prediction_sets(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Boolean array (n, n_classes): class y is in the set if 1 - p(y) <= threshold of y."""
    return (1 - probs) <= thresholds[np.newaxis, :]
