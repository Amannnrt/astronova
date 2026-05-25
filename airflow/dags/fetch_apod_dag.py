"""
DAG 1 — Daily APOD fetch
Runs every day, fetches NASA APOD image, saves it to data/raw,
logs it to image_log.csv
"""
import csv
import io
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import requests
from airflow import DAG
from airflow.operators.python import PythonOperator

PROJECT_ROOT = Path("/usr/local/astronova")
sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner":            "astronova",
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
}

#Tasks
def fetch_and_save_apod(**kwargs):
    context = kwargs
    from dotenv import load_dotenv
    load_dotenv(f"{PROJECT_ROOT}/.env")

    from src.data.nasa_client     import NASAClient
    from src.data.label_extractor import LabelExtractor

    client    = NASAClient()
    extractor = LabelExtractor()

    # Try today and fall back up to 10 days
    apod = None
    for days_back in range(0, 10):
        date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        apod = client.get_apod(date=date)
        if apod:
            break

    if not apod:
        logger.warning("No valid APOD image found in last 10 days")
        return

    url   = apod.get("url")
    title = apod.get("title", "unknown")
    date  = apod.get("date", datetime.now().strftime("%Y-%m-%d"))

    logger.info(f"Fetched APOD: {title} ({date})")

    # Extract label
    label = extractor.from_apod(apod)
    if not label:
        logger.warning(f"Could not extract label for APOD: {title} — skipping")
        return

    # Download image
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    image_bytes = response.content

    # Save to data/raw/<label>/
    save_dir = Path(PROJECT_ROOT) / "data" / "raw" / label
    save_dir.mkdir(parents=True, exist_ok=True)

    filename = f"apod_{date}.jpg"
    filepath = save_dir / filename

    if filepath.exists():
        logger.info(f"Already downloaded: {filename}")
        return

    with open(filepath, "wb") as f:
        f.write(image_bytes)

    logger.info(f"Saved {label}/{filename}")

    # Log to image_log.csv
    metadata_path = Path(PROJECT_ROOT) / "data" / "metadata" / "image_log.csv"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metadata_path, "a", newline="") as f:
        csv.writer(f).writerow([
            filename, label, "apod", date, datetime.now().isoformat()
        ])

    # Push to context for next task
    context["ti"].xcom_push(key="new_image_label", value=label)
    context["ti"].xcom_push(key="new_image_date",  value=date)


def check_retrain_threshold(**kwargs):
    context = kwargs
    """Count images added since last retrain. Log count."""
    metadata_path = Path(PROJECT_ROOT) / "data" / "metadata" / "image_log.csv"

    if not metadata_path.exists():
        logger.info("No metadata file found")
        return

    with open(metadata_path, "r") as f:
        rows = list(csv.DictReader(f))

    apod_images = [r for r in rows if r.get("source") == "apod"]
    logger.info(f"Total APOD images collected so far: {len(apod_images)}")
    context["ti"].xcom_push(key="apod_count", value=len(apod_images))


#DAG 

with DAG(
    dag_id="fetch_apod_daily",
    default_args=DEFAULT_ARGS,
    description="Fetch NASA APOD image daily and save to dataset",
    schedule="0 9 * * *",          # every day at 9am
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["astronova", "data"],
) as dag:

    fetch_task = PythonOperator(
        task_id="fetch_and_save_apod",
        python_callable=fetch_and_save_apod,
    )

    check_task = PythonOperator(
        task_id="check_retrain_threshold",
        python_callable=check_retrain_threshold,
    )

    fetch_task >> check_task
