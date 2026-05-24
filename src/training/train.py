"""
Stage 1 training script — no MLflow yet.
Run: python -m src.training.train
"""
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data.preprocessor    import DataPreprocessor
from src.models.resnet_classifier import AstroClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def _train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = correct = total = 0

    for i, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct    += logits.argmax(1).eq(labels).sum().item()
        total      += labels.size(0)

        if i % 10 == 0:
            logger.info(f"  step {i:>4}/{len(loader)}  loss={loss.item():.4f}")

    return total_loss / len(loader), correct / total


@torch.no_grad()
def _val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        total_loss += criterion(logits, labels).item()
        correct    += logits.argmax(1).eq(labels).sum().item()
        total      += labels.size(0)

    return total_loss / len(loader), correct / total


def train(
    raw_data_dir:       str   = "data/raw",
    processed_data_dir: str   = "data/processed",
    model_save_dir:     str   = "models",
    epochs:             int   = 20,
    batch_size:         int   = 16,
    learning_rate:      float = 1e-4,
    weight_decay:       float = 1e-4,
    patience:           int   = 5,
):
    logger.info(f"Device: {DEVICE}")

    #Data
    preprocessor = DataPreprocessor(raw_data_dir, processed_data_dir)
    preprocessor.split_and_copy()
    train_loader, val_loader, _ = preprocessor.get_dataloaders(batch_size)

    #Model
    model = AstroClassifier(num_classes=4).to(DEVICE)
    logger.info(f"Trainable parameters: {model.trainable_params:,}")

    class_counts = [123, 85, 125, 65]  # galaxy, nebula, planet, star_cluster
    class_weights = torch.tensor(
        [1.0 / c for c in class_counts],
        dtype=torch.float
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    #Optimiser / scheduler
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs) #gradually reduces learning rate

    #Training loop
    best_val_acc    = 0.0
    patience_counter = 0
    history          = []

    for epoch in range(1, epochs + 1):
        logger.info(f"\n── Epoch {epoch}/{epochs} ──────────────────────")

        tr_loss, tr_acc = _train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        va_loss, va_acc = _val_epoch(model, val_loader,   criterion, DEVICE)
        scheduler.step()

        logger.info(f"  train  loss={tr_loss:.4f}  acc={tr_acc:.4f}")
        logger.info(f"  val    loss={va_loss:.4f}  acc={va_acc:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": round(tr_loss, 4), "train_acc": round(tr_acc, 4),
            "val_loss":   round(va_loss, 4), "val_acc":   round(va_acc, 4),
        })

        if va_acc > best_val_acc:
            best_val_acc    = va_acc
            patience_counter = 0
            model.save(f"{model_save_dir}/best_model.pt")
            logger.info(f" new best saved (val_acc={va_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    # Persist metrics
    Path("metrics.json").write_text(json.dumps(history, indent=2))
    logger.info(f"\nDone. Best val accuracy: {best_val_acc:.4f}")
    return best_val_acc


if __name__ == "__main__":
    train()
