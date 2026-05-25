"""
DAG 2 — Monthly retraining
Runs on the 1st of every month.
Triggers dvc repro which runs preprocess + train + logs to MLflow.
"""
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path 

from airflow import DAG
from airflow.operators.python import PythonOperator

PROJECT_ROOT = Path("/usr/local/astronova")

sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner":            "astronova",
    "retries":          1,
    "retry_delay":      timedelta(minutes=10),
    "email_on_failure": False,
}


#Tasks
def update_data_version():
    """Run dvc add and dvc push to version new APOD images."""
    logger.info("Versioning new data with DVC...")

    result = subprocess.run(
        ["dvc", "add", "data/raw"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    logger.info(result.stdout)
    if result.returncode != 0:
        raise Exception(f"dvc add failed: {result.stderr}")

    result = subprocess.run(
        ["dvc", "push"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    logger.info(result.stdout)
    if result.returncode != 0:
        raise Exception(f"dvc push failed: {result.stderr}")

    logger.info("Data versioned and pushed to DagsHub")


def run_training_pipeline():
    """Run dvc repro to trigger preprocess + train."""
    logger.info("Starting retraining pipeline via dvc repro...")

    result = subprocess.run(
        ["dvc", "repro"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
    )
    logger.info(result.stdout)
    if result.returncode != 0:
        raise Exception(f"dvc repro failed: {result.stderr}")

    logger.info("Retraining pipeline completed successfully")


def evaluate_new_model():
    """Run evaluate.py and log results to MLflow."""
    logger.info("Evaluating new model...")

    result = subprocess.run(
        ["python", "-m", "src.training.evaluate"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
    )
    logger.info(result.stdout)
    if result.returncode != 0:
        raise Exception(f"Evaluation failed: {result.stderr}")

    logger.info("Evaluation complete — results logged to MLflow")


def compare_models():
    """
    Compare new model vs current best using MLflow.
    Promotes new model if it beats current production model.
    """
    from dotenv import load_dotenv
    load_dotenv(f"{PROJECT_ROOT}/.env")

    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    client = MlflowClient()

    model_name = "AstroNova-Classifier"

    # Get all versions
    versions = client.search_model_versions(f"name='{model_name}'")
    if len(versions) < 2:
        logger.info("Not enough model versions to compare — keeping current")
        return

    # Sort by version number, get latest two
    versions_sorted = sorted(versions, key=lambda v: int(v.version), reverse=True)
    new_version  = versions_sorted[0]
    prod_version = versions_sorted[1]

    # Get test accuracy from run metrics
    def get_accuracy(run_id):
        run = client.get_run(run_id)
        return run.data.metrics.get("best_val_acc", 0.0)

    new_acc  = get_accuracy(new_version.run_id)
    prod_acc = get_accuracy(prod_version.run_id)

    logger.info(f"New model accuracy:  {new_acc:.4f}")
    logger.info(f"Prod model accuracy: {prod_acc:.4f}")

    if new_acc > prod_acc:
        client.set_model_version_tag(
            model_name, new_version.version, "stage", "production"
        )
        client.set_model_version_tag(
            model_name, prod_version.version, "stage", "archived"
        )
        logger.info(f"New model promoted to production (v{new_version.version})")
    else:
        logger.info("Current model retained — new model did not improve")


#DAG
with DAG(
    dag_id="retrain_monthly",
    default_args=DEFAULT_ARGS,
    description="Monthly retraining pipeline — version data, retrain, evaluate, compare",
    schedule="0 2 1 * *",         # 1st of every month at 2am
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["astronova", "training"],
) as dag:

    version_data = PythonOperator(
        task_id="update_data_version",
        python_callable=update_data_version,
    )

    retrain = PythonOperator(
        task_id="run_training_pipeline",
        python_callable=run_training_pipeline,
    )

    evaluate = PythonOperator(
        task_id="evaluate_new_model",
        python_callable=evaluate_new_model,
    )

    compare = PythonOperator(
        task_id="compare_models",
        python_callable=compare_models,
    )

    version_data >> retrain >> evaluate >> compare
