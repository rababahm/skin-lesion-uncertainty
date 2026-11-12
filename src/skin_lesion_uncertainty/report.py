"""Summarise the model comparisons logged in MLflow: best validation balanced accuracy over seeds.

Run from the repository root:
    uv run python -m skin_lesion_uncertainty.report
"""

import mlflow
import pandas as pd

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "skin-lesion-cnn"
METRIC = "metrics.best_val_balanced_accuracy"
# Settings shared by every run in the comparison; MLflow stores parameters as strings.
SHARED = {"batch_size": "64", "dropout": "0.3", "patience": "5"}
LR = {"cnn": "0.001", "resnet18": "0.0003"}  # lower rate for the pretrained network
KEYS = ["params.arch", "params.size", "params.class_weights"]


def load_runs() -> pd.DataFrame:
    """Finished runs of the experiment, newest first, one row per run."""
    mlflow.set_tracking_uri(TRACKING_URI)
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT])
    runs = runs[runs["status"] == "FINISHED"].copy()
    # Runs from before --arch and --size existed were all the CNN at 28 pixels.
    runs["params.arch"] = runs["params.arch"].fillna("cnn")
    runs["params.size"] = runs["params.size"].fillna("28")
    return runs


def model_summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean, sd and count of the best validation balanced accuracy, per model setting."""
    for name, value in SHARED.items():
        runs = runs[runs[f"params.{name}"] == value]
    runs = runs[runs["params.lr"] == runs["params.arch"].map(LR)]
    # Repeated runs of the same setting and seed: keep the newest.
    runs = runs.drop_duplicates(subset=[*KEYS, "params.seed"], keep="first")
    return runs.groupby(KEYS)[METRIC].agg(["mean", "std", "count"]).round(3)


def main() -> None:
    print("Best validation balanced accuracy (sd over seeds):")
    print(model_summary(load_runs()))


if __name__ == "__main__":
    main()
