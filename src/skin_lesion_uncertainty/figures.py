"""Figures for the README: images per class, and example images for every class.

Run with:
    make figures

Saves:
    reports/figures/class_counts.png   training images per class, with counts and shares
    reports/figures/examples.png       four random training images for each of the 7 classes

Both figures use the training split only. The validation and test splits are not touched.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from medmnist import INFO

from skin_lesion_uncertainty.data import load_split

FIGURE_DIR = Path("reports/figures")
SEED = 0

# Full class names from MedMNIST (keys "0" to "6") and the standard HAM10000 short codes,
# in the same order as the label indices 0 to 6.
CLASS_NAMES = [INFO["dermamnist"]["label"][str(k)] for k in range(7)]
SHORT_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def plot_class_counts(labels: np.ndarray, path: Path) -> None:
    """Horizontal bar chart of training images per class, annotated with count and share."""
    counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    shares = counts / counts.sum()
    order = np.argsort(counts)  # smallest class at the bottom, largest at the top

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh([CLASS_NAMES[k] for k in order], counts[order], color="tab:blue")
    # Write "count (share)" just past the end of each bar.
    for y_pos, k in enumerate(order):
        ax.text(counts[k], y_pos, f"  {counts[k]} ({shares[k]:.1%})", va="center", fontsize=9)
    ax.set_xlabel("Training images")
    ax.set_title("DermaMNIST training split: images per class")
    ax.set_xlim(0, counts.max() * 1.25)  # leave room for the text after the longest bar
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)  # free the memory; matters when a script draws many figures


def plot_examples_per_class(
    images: np.ndarray, labels: np.ndarray, path: Path, per_class: int = 4
) -> None:
    """One row per class with a few random training images, labelled with the short code.

    A purely random grid would be two-thirds nevi (the majority class). One row per class
    shows what every class looks like, including the rare ones.
    """
    rng = np.random.default_rng(SEED)  # fixed seed: the same images on every run
    n_classes = len(CLASS_NAMES)
    fig, axes = plt.subplots(n_classes, per_class, figsize=(per_class * 1.4, n_classes * 1.4))

    for k in range(n_classes):
        # Indices of all training images of class k, then a random few of them.
        idx = rng.choice(np.flatnonzero(labels == k), size=per_class, replace=False)
        for j, i in enumerate(idx):
            ax = axes[k, j]
            ax.imshow(images[i])  # NumPy images are already (H, W, 3): no permute needed
            ax.set_xticks([])
            ax.set_yticks([])
        axes[k, 0].set_ylabel(SHORT_NAMES[k], rotation=0, ha="right", va="center", fontsize=10)

    fig.suptitle("DermaMNIST training images by class (28 × 28)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    images, labels = load_split("train", size=28)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_class_counts(labels, FIGURE_DIR / "class_counts.png")
    plot_examples_per_class(images, labels, FIGURE_DIR / "examples.png")
    print(f"Saved class_counts.png and examples.png in {FIGURE_DIR}/")


if __name__ == "__main__":
    main()
