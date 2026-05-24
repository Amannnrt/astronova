import csv
import logging
import time
import requests
from datetime import datetime
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)

CLASSES = [
    "galaxy", "nebula", "planet", "star_cluster",
    ]


class ImageDownloader:
    def __init__(self,raw_data_dir: str,metadata_path:str):
        self.raw_dir = Path(raw_data_dir)
        self.metadata_path = Path(metadata_path)
        self._init_dirs()
        self._init_metadata()


    def _init_dirs(self):
        for cls in CLASSES:
            (self.raw_dir / cls).mkdir(parents=True, exist_ok=True)

    def _init_metadata(self):
        if not self.metadata_path.exists():
            self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.metadata_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["filename", "class", "nasa_id", "url", "date_fetched"])


    def _already_downloaded(self, nasa_id: str) -> bool:
        with open(self.metadata_path, "r") as f:
            return any(row["nasa_id"] == nasa_id for row in csv.DictReader(f))

    def _log_metadata(self, filename: str, label: str, nasa_id: str, url: str):
        with open(self.metadata_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [filename, label, nasa_id, url, datetime.now().isoformat()]
            )

    def _download(self, url: str, dest: Path, retries: int = 3) -> bool:
        for attempt in range(retries):
            try:
                resp = requests.get(url, timeout=30, stream=True)
                resp.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in resp.iter_content(8192):
                        f.write(chunk)
                return True
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed ({url}): {e}")
                time.sleep(2 ** attempt)
        return False

    def save(self, url: str, label: str, nasa_id: str) -> bool:
        """Download one image and record it in metadata. Returns True on success."""
        if self._already_downloaded(nasa_id):
            logger.debug(f"Skip {nasa_id} — already exists")
            return False

        ext      = url.rsplit(".", 1)[-1].lower()
        ext      = ext if ext in {"jpg", "jpeg", "png"} else "jpg"
        filename = f"{nasa_id}.{ext}"
        dest     = self.raw_dir / label / filename

        if self._download(url, dest):
            self._log_metadata(filename, label, nasa_id, url)
            logger.info(f"Saved  {label}/{filename}")
            return True

        return False
