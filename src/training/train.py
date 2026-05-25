"""
Stage 2 training script — with MLflow tracking + artifacts.
Run: python -m src.training.train
"""

import json
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from dotenv import load_dotenv
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data.preprocessor import DataPreprocessor
from src.models.resnet_classifier import AstroClassifier

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _train_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for i, (imgs, labels) in enumerate(loader):

        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
        if i % 10 == 0:
            logger.info(
                f"step {i:>4}/{len(loader)} | loss={loss.item():.4f}"
            )

    avg_loss = total_loss / len(loader)
    avg_acc = correct / total

    return avg_loss, avg_acc


@torch.no_grad()
def _val_epoch(model, loader, criterion, device):

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        preds = logits.argmax(1)
        correct += preds.eq(labels).sum().item()

        total += labels.size(0)

    avg_loss = total_loss / len(loader)
    avg_acc = correct / total

    return avg_loss, avg_acc


def train(
    raw_data_dir: str = "data/raw",
    processed_data_dir: str = "data/processed",
    model_save_dir: str = "models",
    epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 5,
):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    artifact_root = os.getenv("MLFLOW_ARTIFACT_ROOT", "./mlruns")

    mlflow.set_tracking_uri(tracking_uri)

    # Create experiment with the correct artifact root if it doesn't exist.
    # This prevents MLflow from reusing a stale path stored in the DB.
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    experiment_name = "AstroNova-Classification"
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(
            experiment_name,
            artifact_location=artifact_root,   
        )
        logger.info(f"Created experiment '{experiment_name}' with artifact root: {artifact_root}")
    else:
        logger.info(f"Using existing experiment '{experiment_name}' (artifact location: {experiment.artifact_location})")

    mlflow.set_experiment(experiment_name)

    # ------------------------------------------------------------------ #

    logger.info(f"Using device: {DEVICE}")

    preprocessor = DataPreprocessor(
        raw_data_dir,
        processed_data_dir,
    )
    preprocessor.split_and_copy()

    train_loader, val_loader, test_loader = (
        preprocessor.get_dataloaders(batch_size)
    )

    model = AstroClassifier(num_classes=4).to(DEVICE)

    logger.info(
        f"Trainable parameters: {model.trainable_params:,}"
    )

    class_counts = [123, 85, 125, 65]

    class_weights = torch.tensor(
        [1.0 / c for c in class_counts],
        dtype=torch.float,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )

    Path(model_save_dir).mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name="resnet50_run") as run:

        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "architecture": "resnet50",
            "num_classes": 4,
            "trainable_layers": "layer4 + fc",
            "dropout": 0.5,
            "patience": patience,
            "train_images": len(train_loader.dataset),
            "val_images": len(val_loader.dataset),
        })

        best_val_acc = 0.0
        patience_counter = 0
        history = []

        for epoch in range(1, epochs + 1):

            logger.info(
                f"\n────────── Epoch {epoch}/{epochs} ──────────"
            )

            train_loss, train_acc = _train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                DEVICE,
            )

            val_loss, val_acc = _val_epoch(
                model,
                val_loader,
                criterion,
                DEVICE,
            )

            scheduler.step()

            logger.info(
                f"train loss={train_loss:.4f} | "
                f"train acc={train_acc:.4f}"
            )

            logger.info(
                f"val   loss={val_loss:.4f} | "
                f"val   acc={val_acc:.4f}"
            )

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "learning_rate": scheduler.get_last_lr()[0],
            }, step=epoch)

            history.append({
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 4),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 4),
            })

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0

                best_model_path = (
                    f"{model_save_dir}/best_model.pt"
                )

                model.save(best_model_path)

                logger.info(
                    f"✓ New best model saved "
                    f"(val_acc={val_acc:.4f})"
                )

                mlflow.pytorch.log_model(
                    pytorch_model=model,
                    artifact_path="model",
                    registered_model_name="AstroNova-Classifier",
                )

                mlflow.log_metric(
                    "best_val_acc",
                    best_val_acc,
                )

            else:

                patience_counter += 1

                if patience_counter >= patience:

                    logger.info(
                        f"Early stopping triggered at epoch {epoch}"
                    )

                    break

        epochs_ran = [x["epoch"] for x in history]

        train_losses = [x["train_loss"] for x in history]
        val_losses = [x["val_loss"] for x in history]

        train_accs = [x["train_acc"] for x in history]
        val_accs = [x["val_acc"] for x in history]

        plt.figure(figsize=(8, 5))

        plt.plot(epochs_ran, train_losses, label="Train Loss")
        plt.plot(epochs_ran, val_losses, label="Validation Loss")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.title("Training vs Validation Loss")

        plt.legend()

        loss_plot_path = "loss_curve.png"

        plt.savefig(loss_plot_path)

        plt.close()

        plt.figure(figsize=(8, 5))

        plt.plot(epochs_ran, train_accs, label="Train Accuracy")
        plt.plot(epochs_ran, val_accs, label="Validation Accuracy")

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")

        plt.title("Training vs Validation Accuracy")

        plt.legend()

        acc_plot_path = "accuracy_curve.png"

        plt.savefig(acc_plot_path)

        plt.close()

        metrics_path = "metrics.json"

        Path(metrics_path).write_text(
            json.dumps(history, indent=2)
        )

        mlflow.log_artifact(loss_plot_path)
        mlflow.log_artifact(acc_plot_path)
        mlflow.log_artifact(metrics_path)

        mlflow.log_metrics({
            "final_best_val_acc": best_val_acc,
            "total_epochs_run": epoch,
        })

        logger.info(f"\nRun ID: {run.info.run_id}")

    logger.info(
        f"\nTraining complete. "
        f"Best validation accuracy: {best_val_acc:.4f}"
    )

    return best_val_acc


if __name__ == "__main__":
    train()
