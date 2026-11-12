"""Final evaluation on the test set, following the protocol frozen on 12 November 2026.

Needs artifacts/resnet_final, saved with:
    uv run python -m skin_lesion_uncertainty.train --arch resnet18 --size 64 --lr 3e-4 \
        --seed 0 --save-logits artifacts/resnet_final --save-test-logits

Run from the repository root:
    uv run python -m skin_lesion_uncertainty.final_evaluation
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import log_softmax, softmax
from sklearn.metrics import balanced_accuracy_score

from skin_lesion_uncertainty.conformal import (
    classwise_prediction_sets,
    classwise_thresholds,
    conformal_scores,
    conformal_threshold,
    coverage_and_size,
    partial_classwise_thresholds,
    prediction_sets,
)
from skin_lesion_uncertainty.decision import referral_summary
from skin_lesion_uncertainty.evaluate import expected_calibration_error, fit_temperature, top_label

FOLDER = Path("artifacts/resnet_final")
FIGURE = Path("reports/figures/referral_tradeoff.png")
SHORT = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]  # DermaMNIST classes 0 to 6
NON_BENIGN = (0, 1, 4)  # actinic keratoses, basal cell carcinoma, melanoma
ALPHA = 0.10
ALPHAS = [0.02, 0.05, 0.10, 0.15, 0.20]
N_SPLITS, N_BOOT = 1000, 2000


def load(split: str) -> tuple[np.ndarray, np.ndarray]:
    logits = np.load(FOLDER / f"{split}_logits.npy").astype(np.float64)
    return logits, np.load(FOLDER / f"{split}_labels.npy")


def mean_nll(logits: np.ndarray, labels: np.ndarray) -> float:
    return float(-log_softmax(logits, axis=1)[np.arange(len(labels)), labels].mean())


def ece(logits: np.ndarray, labels: np.ndarray) -> float:
    return expected_calibration_error(*top_label(logits, labels))


def balanced_accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    return balanced_accuracy_score(labels, logits.argmax(axis=1))


def with_interval(metric, logits, labels, rng) -> str:
    """The metric on the test set, with a bootstrap 95% interval over test images."""
    n = len(labels)
    boots = [metric(logits[idx], labels[idx]) for idx in rng.integers(0, n, (N_BOOT, n))]
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return f"{metric(logits, labels):.3f} [{lo:.3f}, {hi:.3f}]"


def make_sets(method, probs, labels, cal, ev, alpha) -> np.ndarray:
    scores = conformal_scores(probs[cal], labels[cal])
    n_classes = probs.shape[1]
    if method == "standard":
        return prediction_sets(probs[ev], conformal_threshold(scores, alpha))
    if method == "mondrian":
        thresholds = classwise_thresholds(scores, labels[cal], n_classes, alpha)
    else:
        thresholds = partial_classwise_thresholds(scores, labels[cal], n_classes, alpha, NON_BENIGN)
    return classwise_prediction_sets(probs[ev], thresholds)


def repeated_splits(method, probs, labels, alpha) -> dict:
    """Coverage, set size and referral over random 50/50 calibration/evaluation splits."""
    rng = np.random.default_rng(0)  # the same splits for every method and alpha
    n, n_classes = probs.shape
    rows, hits, seen = [], np.zeros(n_classes), np.zeros(n_classes)
    for _ in range(N_SPLITS):
        perm = rng.permutation(n)
        cal, ev = perm[: n // 2], perm[n // 2 :]
        sets = make_sets(method, probs, labels, cal, ev, alpha)
        coverage, size = coverage_and_size(sets, labels[ev])
        r = referral_summary(sets, labels[ev])
        rows.append([coverage, size, r["referral_rate"], r["missed_share_of_non_benign"]])
        covered = sets[np.arange(len(ev)), labels[ev]]
        hits += np.bincount(labels[ev], weights=covered, minlength=n_classes)
        seen += np.bincount(labels[ev], minlength=n_classes)
    rows = np.array(rows)
    return {
        "coverage": rows[:, 0].mean(),
        "coverage_range": np.quantile(rows[:, 0], [0.025, 0.975]),
        "size": rows[:, 1].mean(),
        "referral": rows[:, 2].mean(),
        "missed": rows[:, 3].mean(),
        "per_class": hits / seen,
    }


def main() -> None:
    val_logits, val_labels = load("val")
    test_logits, test_labels = load("test")
    t = fit_temperature(val_logits, val_labels)  # validation only: the test set fits nothing
    scaled = test_logits / t
    rng = np.random.default_rng(0)

    print(f"Test set: {len(test_labels)} images; T = {t:.2f} (fitted on validation)\n")
    bal_acc = with_interval(balanced_accuracy, test_logits, test_labels, rng)
    ece_before = with_interval(ece, test_logits, test_labels, rng)
    ece_after = with_interval(ece, scaled, test_labels, rng)
    nll_before = with_interval(mean_nll, test_logits, test_labels, rng)
    nll_after = with_interval(mean_nll, scaled, test_labels, rng)
    print("| Metric (95% bootstrap interval) | Before scaling | After scaling |")
    print("| --- | --- | --- |")
    print(f"| Balanced accuracy | {bal_acc} | unchanged |")
    print(f"| ECE | {ece_before} | {ece_after} |")
    print(f"| NLL | {nll_before} | {nll_after} |")

    probs = softmax(scaled, axis=1)
    print(f"\nConformal sets, alpha = {ALPHA}, {N_SPLITS} random 50/50 splits of the test set\n")
    print(
        "| Method | Coverage (95% of splits) | Melanoma | Set size | Referred | Missed non-benign |"
    )
    print("| --- | --- | --- | --- | --- | --- |")
    results = {}
    for method in ["standard", "mondrian", "hybrid"]:
        r = results[method] = repeated_splits(method, probs, test_labels, ALPHA)
        lo, hi = r["coverage_range"]
        print(
            f"| {method} | {r['coverage']:.3f} ({lo:.3f} to {hi:.3f}) | {r['per_class'][4]:.2f} | "
            f"{r['size']:.2f} | {r['referral']:.0%} | {r['missed']:.1%} |"
        )
    print("\nPer-class coverage:")
    for method, r in results.items():
        cells = "  ".join(f"{s} {c:.2f}" for s, c in zip(SHORT, r["per_class"], strict=True))
        print(f"  {method:9s} {cells}")

    print("\n| alpha | Method | Coverage | Set size | Referred | Missed non-benign |")
    print("| --- | --- | --- | --- | --- | --- |")
    curves = {"standard": [], "hybrid": []}
    for alpha in ALPHAS:
        for method, points in curves.items():
            r = repeated_splits(method, probs, test_labels, alpha)
            points.append((100 * r["referral"], 100 * r["missed"]))
            print(
                f"| {alpha:.2f} | {method} | {r['coverage']:.3f} | {r['size']:.2f} | "
                f"{r['referral']:.0%} | {r['missed']:.1%} |"
            )

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for method, points in curves.items():
        referral, missed = zip(*points, strict=True)
        ax.plot(referral, missed, "o-", label=method)
        for alpha, x, y in zip(ALPHAS, referral, missed, strict=True):
            ax.annotate(f"alpha {alpha:.2f}", (x, y), textcoords="offset points", xytext=(5, 3))
    ax.set(
        xlabel="Referred to a clinician (%)",
        ylabel="Non-benign lesions auto-accepted (%)",
        title="Referral trade-off on the test set",
    )
    ax.legend()
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=150)
    print(f"\nSaved {FIGURE}")


if __name__ == "__main__":
    main()
