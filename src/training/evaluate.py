import logging
import os

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import seaborn as sns
import torch
from dotenv import load_dotenv
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.data.preprocessor import DataPreprocessor
from src.models.resnet_classifier import AstroClassifier

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)

logger = logging.getLogger(__name__)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = [
    "galaxy",
    "nebula",
    "planet",
    "star_cluster",
]


@torch.no_grad()
def evaluate(
    model_path: str = "models/best_model.pt",
    raw_data_dir: str = "data/raw",
    processed_data_dir: str = "data/processed",
    batch_size: int = 16,
):


    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    )

    mlflow.set_experiment("AstroNova-Classification")

    model = AstroClassifier.load(
        model_path,
        device=str(DEVICE),
    )

    model.to(DEVICE)

    model.eval()

    logger.info(f"Loaded model from: {model_path}")

    preprocessor = DataPreprocessor(
        raw_data_dir,
        processed_data_dir,
    )

    _, _, test_loader = (
        preprocessor.get_dataloaders(batch_size)
    )


    all_preds = []
    all_labels = []
    all_confs = []

    for imgs, labels in test_loader:

        imgs = imgs.to(DEVICE)

        logits = model(imgs)

        probs = torch.softmax(logits, dim=1)

        confs, preds = probs.max(dim=1)

        all_preds.extend(preds.cpu().numpy())

        all_labels.extend(labels.numpy())

        all_confs.extend(confs.cpu().numpy())


    acc = np.mean(
        np.array(all_preds) == np.array(all_labels)
    )

    f1 = f1_score(
        all_labels,
        all_preds,
        average="weighted",
    )

    avg_conf = float(np.mean(all_confs))

    cm = confusion_matrix(
        all_labels,
        all_preds,
    )

    report = classification_report(
        all_labels,
        all_preds,
        target_names=CLASS_NAMES,
    )

    logger.info(f"\n{report}")

    logger.info(f"\nConfusion Matrix:\n{cm}")

    logger.info(
        f"\nAverage confidence: {avg_conf:.4f}"
    )


    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    cm_plot_path = "confusion_matrix.png"
    plt.savefig(cm_plot_path)

    plt.close()
    plt.figure(figsize=(8, 5))
    plt.hist(all_confs, bins=20)
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.title("Prediction Confidence Distribution")
    conf_plot_path = "confidence_distribution.png"
    plt.savefig(conf_plot_path)
    plt.close()


    cm_csv_path = "confusion_matrix.csv"

    np.savetxt(
        cm_csv_path,
        cm,
        delimiter=",",
        fmt="%d",
    )

    report_path = "classification_report.txt"

    with open(report_path, "w") as f:
        f.write(report)

    with mlflow.start_run(run_name="evaluation"):


        mlflow.log_metrics({
            "test_accuracy": round(acc, 4),
            "test_f1_weighted": round(f1, 4),
            "avg_confidence": round(avg_conf, 4),
        })

        f1_per_class = f1_score(
            all_labels,
            all_preds,
            average=None,
        )

        for cls_name, score in zip(
            CLASS_NAMES,
            f1_per_class,
        ):

            mlflow.log_metric(
                f"f1_{cls_name}",
                round(float(score), 4),
            )

        mlflow.log_artifact(cm_plot_path)
        mlflow.log_artifact(conf_plot_path)
        mlflow.log_artifact(cm_csv_path)
        mlflow.log_artifact(report_path)
        logger.info(f"\nTest Accuracy : {acc:.4f}")
        logger.info(f"Weighted F1   : {f1:.4f}")
        logger.info(
            f"Avg Confidence: {avg_conf:.4f}"
        )

    return acc, f1, avg_conf


if __name__ == "__main__":
    evaluate()
