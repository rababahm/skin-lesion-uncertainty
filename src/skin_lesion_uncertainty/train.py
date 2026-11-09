"""Train SmallCNN on 28 x 28 DermaMNIST, with early stopping on validation balanced accuracy.

Every run is logged to MLflow: the settings, the metrics after each epoch, and the best model.

Run with:
    make train
    uv run python -m skin_lesion_uncertainty.train --lr 3e-4 --no-class-weights
    uv run python -m skin_lesion_uncertainty.train --save-logits artifacts/cw_on_seed0

Compare runs in the browser with:
    uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import argparse
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch import nn

from skin_lesion_uncertainty.data import class_weights, make_loaders
from skin_lesion_uncertainty.evaluate import collect_logits
from skin_lesion_uncertainty.model import SmallCNN

MODEL_DIR = Path("models")  # *.pt files are ignored by Git
TRACKING_URI = "sqlite:///mlflow.db"  # local MLflow database, ignored by Git
EXPERIMENT = "skin-lesion-cnn"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SmallCNN on DermaMNIST.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5, help="epochs without improvement")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-class-weights",
        dest="class_weights",
        action="store_false",
        help="use an unweighted loss, for comparison",
    )
    parser.add_argument(
        "--save-logits",
        type=Path,
        default=None,
        help="folder for the best model's validation logits and labels",
    )
    return parser.parse_args()


def pick_device() -> str:
    if torch.accelerator.is_available():
        return torch.accelerator.current_accelerator().type
    return "cpu"


def train_one_epoch(model, loader, loss_fn, optimizer, device) -> float:
    """One pass over the training data; returns the average training loss."""
    model.train()  # dropout on
    summed_loss, num_images = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        loss = loss_fn(model(images), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        summed_loss += loss.item() * len(images)
        num_images += len(images)
    return summed_loss / num_images


def evaluate(model, loader, loss_fn, device) -> dict[str, float]:
    """Loss, balanced accuracy and macro-F1 on one split, with no learning."""
    model.eval()  # dropout off: every unit is used, deterministically
    summed_loss = 0.0
    predictions, targets = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            summed_loss += loss_fn(logits, labels).item() * len(images)
            predictions.append(logits.argmax(dim=1).cpu())
            targets.append(labels.cpu())
    y_pred = torch.cat(predictions).numpy()
    y_true = torch.cat(targets).numpy()
    return {
        "loss": summed_loss / len(y_true),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)  # same initial weights on every run
    device = pick_device()

    loaders = make_loaders(batch_size=args.batch_size, seed=args.seed)
    train_labels = loaders["train"].dataset.tensors[1].numpy()
    weights = class_weights(train_labels).to(device) if args.class_weights else None
    loss_fn = nn.CrossEntropyLoss(weight=weights)  # weight=None means unweighted

    model = SmallCNN(dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    run_name = f"cnn-lr{args.lr:g}-{'weighted' if args.class_weights else 'unweighted'}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(vars(args) | {"device": device, "n_params": n_params})

        best_score, best_epoch, best_state = -1.0, 0, None
        for epoch in range(1, args.max_epochs + 1):
            start = time.perf_counter()
            train_loss = train_one_epoch(model, loaders["train"], loss_fn, optimizer, device)
            val = evaluate(model, loaders["val"], loss_fn, device)
            metrics = {"train_loss": train_loss} | {f"val_{k}": v for k, v in val.items()}
            mlflow.log_metrics(metrics, step=epoch)
            print(
                f"Epoch {epoch:2d}: train loss {train_loss:.4f} | "
                f"val loss {val['loss']:.4f} | "
                f"val balanced acc {val['balanced_accuracy']:.3f} | "
                f"{time.perf_counter() - start:.1f} s"
            )

            # Keep the best model; stop after `patience` epochs without improvement.
            if val["balanced_accuracy"] > best_score:
                best_score, best_epoch = val["balanced_accuracy"], epoch
                # Copy to the CPU now: later epochs keep changing the model's own tensors.
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            elif epoch - best_epoch >= args.patience:
                print(f"Stopping: no improvement for {args.patience} epochs.")
                break

        MODEL_DIR.mkdir(exist_ok=True)
        model_path = MODEL_DIR / "cnn_best.pt"
        torch.save(best_state, model_path)
        mlflow.log_artifact(str(model_path))
        mlflow.log_metrics({"best_epoch": best_epoch, "best_val_balanced_accuracy": best_score})

        model.load_state_dict(best_state)  # the best epoch's weights, not the last epoch's
        if args.save_logits is not None:
            logits, labels = collect_logits(model, loaders["val"], device)
            args.save_logits.mkdir(parents=True, exist_ok=True)
            np.save(args.save_logits / "val_logits.npy", logits)
            np.save(args.save_logits / "val_labels.npy", labels)
            print(f"Saved validation logits {logits.shape} to {args.save_logits}")

        print(f"\nBest validation balanced accuracy {best_score:.3f} at epoch {best_epoch}")
        print(f"Saved {model_path}; target to beat (logistic regression): 0.429")


if __name__ == "__main__":
    main()
