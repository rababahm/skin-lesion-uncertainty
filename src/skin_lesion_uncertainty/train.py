"""Train a classifier on DermaMNIST, with early stopping on validation balanced accuracy.

Two architectures: a small CNN on 28 x 28 images, and an ImageNet-pretrained ResNet-18
fine-tuned on 64 x 64 images. Every run is logged to MLflow: the settings, the metrics
after each epoch, and the best model.

Run with:
    make train
    uv run python -m skin_lesion_uncertainty.train --lr 3e-4 --no-class-weights
    uv run python -m skin_lesion_uncertainty.train --arch resnet18 --size 64 --lr 3e-4
    uv run python -m skin_lesion_uncertainty.train --save-logits artifacts/cw_on_seed0

Compare runs in the browser with:
    uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import argparse
from pathlib import Path

import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch import nn

from skin_lesion_uncertainty.model import SmallCNN, pretrained_resnet18

MODEL_DIR = Path("models")  # *.pt files are ignored by Git
TRACKING_URI = "sqlite:///mlflow.db"  # local MLflow database, ignored by Git
EXPERIMENT = "skin-lesion-cnn"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a classifier on DermaMNIST.")
    parser.add_argument("--arch", choices=["cnn", "resnet18"], default="cnn")
    parser.add_argument("--size", type=int, choices=[28, 64], default=28, help="image size")
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
    args = parser.parse_args()
    if args.arch == "cnn" and args.size != 28:
        parser.error("the small CNN is built for 28 x 28 images; use --size 28")
    return args


def build_model(args: argparse.Namespace) -> nn.Module:
    if args.arch == "resnet18":
        return pretrained_resnet18(n_classes=7)
    return SmallCNN(dropout=args.dropout)


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
    torch.manual_seed(args.seed)  # same initial weights on
