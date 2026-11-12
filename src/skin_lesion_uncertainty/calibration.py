"""Temperature scaling and reliability diagrams for the saved validation logits.

Needs the logits from train.py --save-logits (resnet_cw_on_seed0, cw_on_seed0, cw_off_seed0).

Run from the repository root:
    uv run python -m skin_lesion_uncertainty.calibration
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import log_softmax
from sklearn.metrics import balanced_accuracy_score

from skin_lesion_uncertainty.evaluate import (
    expected_calibration_error,
    fit_temperature,
    reliability_bins,
    top_label,
)

ARTIFACTS = Path("artifacts")  # ignored by Git
FIGURE = Path("reports/figures/reliability_validation.png")
MODELS = {
    "ResNet-18, 64 px": "resnet_cw_on_seed0",
    "CNN, class weights on": "cw_on_seed0",
    "CNN, class weights off": "cw_off_seed0",
}
HEADER = ["Model", "Balanced acc.", "T", "ECE before", "ECE after", "NLL before", "NLL after"]


def mean_nll(logits: np.ndarray, labels: np.ndarray) -> float:
    return float(-log_softmax(logits, axis=1)[np.arange(len(labels)), labels].mean())


def ece(logits: np.ndarray, labels: np.ndarray) -> float:
    return expected_calibration_error(*top_label(logits, labels))


def plot_reliability(ax, logits, labels, label, style) -> None:
    counts, mean_confidence, accuracy = reliability_bins(*top_label(logits, labels))
    keep = counts >= 10  # bins with fewer images are too noisy to read
    ax.plot(mean_confidence[keep], accuracy[keep], style, label=label)


def main() -> None:
    fig, axes = plt.subplots(1, len(MODELS), figsize=(15, 4.5), sharey=True)
    print("| " + " | ".join(HEADER) + " |")
    print("|" + " --- |" * len(HEADER))

    for ax, (name, folder) in zip(axes, MODELS.items(), strict=True):
        logits = np.load(ARTIFACTS / folder / "val_logits.npy").astype(np.float64)
        labels = np.load(ARTIFACTS / folder / "val_labels.npy")
        t = fit_temperature(logits, labels)
        scaled = logits / t
        row = [
            name,
            f"{balanced_accuracy_score(labels, logits.argmax(axis=1)):.3f}",
            f"{t:.2f}",
            f"{ece(logits, labels):.3f}",
            f"{ece(scaled, labels):.3f}",
            f"{mean_nll(logits, labels):.3f}",
            f"{mean_nll(scaled, labels):.3f}",
        ]
        print("| " + " | ".join(row) + " |")

        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
        plot_reliability(ax, logits, labels, f"Before (ECE {row[3]})", "o-")
        plot_reliability(ax, scaled, labels, f"After, T = {row[2]} (ECE {row[4]})", "s-")
        ax.set(title=name, xlabel="Mean confidence in bin", xlim=(0, 1), ylim=(0, 1))
        ax.legend(loc="upper left")

    axes[0].set_ylabel("Accuracy in bin")
    fig.suptitle(
        "Validation reliability, seed 0, bins with at least 10 images (T fitted on the same data)"
    )
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=150)
    print(f"\nSaved {FIGURE}")


if __name__ == "__main__":
    main()
