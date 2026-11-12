"""Stress test: how accuracy, calibration and coverage change when test images are shifted.

Conformal thresholds are calibrated on clean test images (one random half) and applied to
shifted images (the other half), which breaks exchangeability on purpose.

Needs models/resnet18_best.pt and artifacts/resnet_final from the frozen run (seed 0).

Run from the repository root:
    uv run python -m skin_lesion_uncertainty.shift
"""

from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms.v2 import functional as F

from skin_lesion_uncertainty.conformal import (
    classwise_prediction_sets,
    conformal_scores,
    conformal_threshold,
    coverage_and_size,
    partial_classwise_thresholds,
    prediction_sets,
)
from skin_lesion_uncertainty.data import make_loaders
from skin_lesion_uncertainty.evaluate import collect_logits, fit_temperature
from skin_lesion_uncertainty.final_evaluation import ALPHA, NON_BENIGN, ece, load
from skin_lesion_uncertainty.model import pretrained_resnet18
from skin_lesion_uncertainty.train import pick_device

MODEL_PATH = Path("models/resnet18_best.pt")
N_SPLITS = 500
MELANOMA = 4


def low_resolution(pixels: int):
    """Downsample to pixels x pixels, then back up to 64 x 64."""
    return lambda x: F.resize(F.resize(x, [pixels, pixels], antialias=True), [64, 64])


SHIFTS = [
    ("none", "", lambda x: x),
    ("blur", "sigma 0.5", lambda x: F.gaussian_blur(x, kernel_size=9, sigma=0.5)),
    ("blur", "sigma 1", lambda x: F.gaussian_blur(x, kernel_size=9, sigma=1.0)),
    ("blur", "sigma 2", lambda x: F.gaussian_blur(x, kernel_size=9, sigma=2.0)),
    ("brightness", "x 0.5", lambda x: F.adjust_brightness(x, 0.5)),
    ("brightness", "x 0.75", lambda x: F.adjust_brightness(x, 0.75)),
    ("brightness", "x 1.25", lambda x: F.adjust_brightness(x, 1.25)),
    ("brightness", "x 1.5", lambda x: F.adjust_brightness(x, 1.5)),
    ("resolution", "32 px", low_resolution(32)),
    ("resolution", "16 px", low_resolution(16)),
]


def logits_for(model, images: torch.Tensor, labels: torch.Tensor, device: str) -> np.ndarray:
    loader = DataLoader(TensorDataset(images, labels), batch_size=256)
    logits, _ = collect_logits(model, loader, device)
    return logits.astype(np.float64)


def coverage_under_shift(clean_probs, shifted_probs, labels, method) -> np.ndarray:
    """Mean coverage, set size and melanoma coverage: calibrate clean, evaluate shifted."""
    rng = np.random.default_rng(0)
    n, n_classes = clean_probs.shape
    rows = []
    for _ in range(N_SPLITS):
        perm = rng.permutation(n)
        cal, ev = perm[: n // 2], perm[n // 2 :]
        scores = conformal_scores(clean_probs[cal], labels[cal])
        if method == "standard":
            sets = prediction_sets(shifted_probs[ev], conformal_threshold(scores, ALPHA))
        else:
            thresholds = partial_classwise_thresholds(
                scores, labels[cal], n_classes, ALPHA, NON_BENIGN
            )
            sets = classwise_prediction_sets(shifted_probs[ev], thresholds)
        coverage, size = coverage_and_size(sets, labels[ev])
        covered = sets[np.arange(len(ev)), labels[ev]]
        rows.append([coverage, size, covered[labels[ev] == MELANOMA].mean()])
    return np.array(rows).mean(axis=0)


def main() -> None:
    device = pick_device()
    val_logits, val_labels = load("val")
    saved_test_logits, test_labels = load("test")
    t = fit_temperature(val_logits, val_labels)  # the same T as the final evaluation

    model = pretrained_resnet18(n_classes=7, pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.to(device)

    images, labels = make_loaders(batch_size=256, size=64, seed=0)["test"].dataset.tensors
    clean_logits = logits_for(model, images, labels, device)
    if not np.allclose(clean_logits, saved_test_logits, atol=1e-3):
        raise SystemExit(f"{MODEL_PATH} is not the frozen model: re-run seed 0 first.")
    clean_probs = softmax(clean_logits / t, axis=1)

    print(f"Calibrated on clean test images, evaluated on shifted ones; alpha = {ALPHA}\n")
    print(
        "| Shift | Level | Balanced acc. | ECE after T | Coverage, standard | "
        "Coverage, hybrid | Melanoma, hybrid | Set size, hybrid |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for name, level, shift in SHIFTS:
        logits = logits_for(model, shift(images), labels, device)
        probs = softmax(logits / t, axis=1)
        std_cov, _, _ = coverage_under_shift(clean_probs, probs, test_labels, "standard")
        hyb_cov, hyb_size, hyb_mel = coverage_under_shift(clean_probs, probs, test_labels, "hybrid")
        bal_acc = balanced_accuracy_score(test_labels, logits.argmax(axis=1))
        print(
            f"| {name} | {level} | {bal_acc:.3f} | {ece(logits / t, test_labels):.3f} | "
            f"{std_cov:.3f} | {hyb_cov:.3f} | {hyb_mel:.2f} | {hyb_size:.2f} |"
        )


if __name__ == "__main__":
    main()
