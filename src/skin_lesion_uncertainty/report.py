"""Summarise the class-weights comparison logged in MLflow.

Run from the repository root:
    uv run python -m skin_lesion_uncertainty.report
"""

import mlflow
import pandas as pd

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "skin-lesion-cnn"
METRIC = "metrics.best_val_balanced_accuracy"
SETTING = "params.class_weights"
SEED = "params.seed"
# Only runs with the default settings belong in this comparison.
# MLflow stores every parameter as a string.
DEFAULTS = {"lr": "0.001", "batch_size": "64", "dropout": "0.3", "patience": "5"}


def load_runs() -> pd.DataFrame:
    """All runs of the experiment, newest first, one row per run."""
    mlflow.set_tracking_uri(TRACKING_URI)
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT])
    needed = {METRIC, SETTING, SEED} | {f"params.{name}" for name in DEFAULTS}
    missing = needed - set(runs.columns)
    if missing:
        raise SystemExit(f"Missing columns {missing}.\nAvailable: {list(runs.columns)}")
    return runs


def class_weight_summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean, sd and count of the best validation balanced accuracy, per setting."""
    runs = runs[runs["status"] == "FINISHED"]
    for name, value in DEFAULTS.items():
        runs = runs[runs[f"params.{name}"] == value]
    # The loop ran twice: keep one run per setting and seed (the newest).
    runs = runs.drop_duplicates(subset=[SETTING, SEED], keep="first")
    return runs.groupby(SETTING)[METRIC].agg(["mean", "std", "count"]).round(3)


def main() -> None:
    summary = class_weight_summary(load_runs())
    print("Best validation balanced accuracy, by class weights (sd over seeds):")
    print(summary)


if __name__ == "__main__":
    main()
