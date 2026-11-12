"""Referral rule on top of conformal prediction sets."""

import numpy as np

# DermaMNIST class indices, as in medmnist.INFO["dermamnist"]["label"].
BENIGN = (2, 3, 5, 6)  # benign keratosis-like, dermatofibroma, melanocytic nevi, vascular


def auto_accept(sets: np.ndarray, benign: tuple[int, ...] = BENIGN) -> np.ndarray:
    """True where the set holds exactly one class and that class is benign."""
    single = sets.sum(axis=1) == 1
    benign_only = sets[:, list(benign)].sum(axis=1) == 1
    return single & benign_only


def referral_summary(
    sets: np.ndarray, labels: np.ndarray, benign: tuple[int, ...] = BENIGN
) -> dict[str, float]:
    """Referral rate, and how often a non-benign lesion is auto-accepted."""
    accepted = auto_accept(sets, benign)
    not_benign = ~np.isin(labels, benign)
    missed = accepted & not_benign
    return {
        "referral_rate": float(1 - accepted.mean()),
        "missed_rate": float(missed.mean()),  # share of all images
        "missed_share_of_non_benign": float(missed.sum() / max(not_benign.sum(), 1)),
    }
