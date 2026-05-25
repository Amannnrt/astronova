
"""
Drift detector — monitors prediction confidence over time.
When average confidence drops below threshold, flags drift
and triggers Airflow retraining DAG.
"""
import logging
import os
import requests
from collections import deque
from datetime import datetime
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class DriftDetector:
    def __init__(
        self,
        window_size: int = 100,
        threshold: float = 0.75,
        log_path: str = "data/metadata/drift_log.json",
    ):
        self.window    = deque(maxlen=window_size)
        self.threshold = threshold
        self.log_path  = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._drift_triggered = False

    def log_confidence(self, confidence: float, predicted_class: str):
        self.window.append({
            "confidence":      confidence,
            "predicted_class": predicted_class,
            "timestamp":       datetime.now().isoformat(),
        })

    @property
    def avg_confidence(self) -> float:
        if not self.window:
            return 1.0
        return sum(w["confidence"] for w in self.window) / len(self.window)

    @property
    def is_drifting(self) -> bool:
        if len(self.window) < self.window.maxlen:
            return False
        return self.avg_confidence < self.threshold

    def check_and_trigger(self):
        """Check drift and trigger Airflow DAG if drifting."""
        if not self.is_drifting:
            self._drift_triggered = False
            return

        if self._drift_triggered:
            logger.info("Drift already triggered — waiting for retraining")
            return

        logger.warning(
            f"Drift detected — avg confidence {self.avg_confidence:.4f} "
            f"below threshold {self.threshold}"
        )

        self._log_drift_event()
        self._trigger_airflow_dag()
        self._drift_triggered = True

    def _log_drift_event(self):
        event = {
            "timestamp":       datetime.now().isoformat(),
            "avg_confidence":  self.avg_confidence,
            "threshold":       self.threshold,
            "window_size":     len(self.window),
        }
        logs = []
        if self.log_path.exists():
            with open(self.log_path) as f:
                logs = json.load(f)
        logs.append(event)
        with open(self.log_path, "w") as f:
            json.dump(logs, f, indent=2)
        logger.info(f"Drift event logged to {self.log_path}")

    def _trigger_airflow_dag(self):
        """Trigger retrain_monthly DAG via Airflow REST API."""
        airflow_url = os.getenv("AIRFLOW_URL", "http://localhost:8080")
        dag_id      = "retrain_monthly"

        try:
            response = requests.post(
                f"{airflow_url}/api/v1/dags/{dag_id}/dagRuns",
                json={"conf": {"trigger_reason": "drift_detected"}},
                auth=(
                    os.getenv("AIRFLOW_USERNAME", "admin"),
                    os.getenv("AIRFLOW_PASSWORD", "admin"),
                ),
                timeout=10,
            )
            if response.status_code == 200:
                logger.info(f"Airflow DAG {dag_id} triggered successfully")
            else:
                logger.warning(f"Failed to trigger DAG: {response.status_code} {response.text}")
        except requests.exceptions.ConnectionError:
            logger.warning("Airflow not reachable — drift logged but DAG not triggered")

    def get_status(self) -> dict:
        return {
            "window_size":     len(self.window),
            "avg_confidence":  round(self.avg_confidence, 4),
            "is_drifting":     self.is_drifting,
            "threshold":       self.threshold,
            "drift_triggered": self._drift_triggered,
        }
