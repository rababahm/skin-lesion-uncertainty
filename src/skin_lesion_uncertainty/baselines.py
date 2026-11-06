"""Baselines for DermaMNIST: majority class, logistic regression and gradient boosting.

Each model is scored with stratified 5-fold cross-validation on the training split
(mean and standard deviation across folds), then fitted on the full training split
and scored once on the validation split. The test split is not touched.

Run with:
    make baselines
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from skin_lesion_uncertainty.data import load_split

SIZE = 28
SEED = 0
N_SPLITS = 5
OUTPUT_PATH = Path("reports/baselines.csv")


def load_flat(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Images flattened to (N, 3 * SIZE * SIZE) float32 in [0, 1], and labels (N,)."""
    images, labels = load_split(split, size=SIZE)
    flat = images.reshape(len(images), -1).astype(np.float32) / 255.0
    return flat, labels


def build_models() -> dict[str, BaseEstimator]:
    """Untrained models. Class weights match the CNN, so the comparison is fair."""
    return {
        "majority": DummyClassifier(strategy="most_frequent"),
        "logistic": Pipeline(
            [
                ("scale", StandardScaler()),  # refitted inside each fold: no leakage
                (
                    "classify",
                    LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced"),
                ),
            ]
        ),
        "boosting": HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED),
    }


def evaluate(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cv: StratifiedKFold,
) -> dict[str, float]:
    """Cross-validated scores on train (mean and std), then one score on validation."""
    cv_scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=["balanced_accuracy", "f1_macro"],
        error_score="raise",
    )
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    return {
        "cv_bal_acc_mean": cv_scores["test_balanced_accuracy"].mean(),
        "cv_bal_acc_std": cv_scores["test_balanced_accuracy"].std(),
        "cv_f1_macro_mean": cv_scores["test_f1_macro"].mean(),
        "cv_f1_macro_std": cv_scores["test_f1_macro"].std(),
        "val_bal_acc": balanced_accuracy_score(y_val, val_pred),
        "val_f1_macro": f1_score(y_val, val_pred, average="macro"),
    }


def mean_pm_std(results: pd.DataFrame, mean_col: str, std_col: str) -> list[str]:
    """Format two numeric columns as 'mean ± std' strings."""
    pairs = zip(results[mean_col], results[std_col], strict=True)
    return [f"{m:.3f} ± {s:.3f}" for m, s in pairs]


def format_table(results: pd.DataFrame) -> str:
    """Readable console table: mean ± std for CV, single numbers for validation."""
    table = pd.DataFrame(
        {
            "model": results["model"],
            "CV balanced accuracy": mean_pm_std(results, "cv_bal_acc_mean", "cv_bal_acc_std"),
            "CV macro-F1": mean_pm_std(results, "cv_f1_macro_mean", "cv_f1_macro_std"),
            "val balanced accuracy": results["val_bal_acc"].map("{:.3f}".format),
            "val macro-F1": results["val_f1_macro"].map("{:.3f}".format),
        }
    )
    return table.to_string(index=False)


def main() -> None:
    X_train, y_train = load_flat("train")
    X_val, y_val = load_flat("val")
    # One splitter shared by every model, so all are scored on the same folds.
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    rows = []
    for name, model in build_models().items():
        print(f"Evaluating {name}...", flush=True)
        rows.append({"model": name, **evaluate(model, X_train, y_train, X_val, y_val, cv)})

    results = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)  # numeric columns, for later analysis

    print()
    print(format_table(results))
    print(f"\nSaved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
