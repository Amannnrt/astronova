"""
One-time script to download the initial training dataset from NASA Image Library.
Run: python scripts/seed_data.py
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.nasa_client    import NASAClient
from src.data.label_extractor import LabelExtractor
from src.data.downloader     import ImageDownloader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# Queries per target class
# Multiple queries increase variety within a class
IMAGES_PER_CLASS = 150      # adjust as needed

"""
QUERIES: dict[str, list[str]] = {
    "galaxy":       ["hubble galaxy", "james webb galaxy", "hubble spiral galaxy"],
    "nebula":       ["hubble nebula", "james webb nebula", "hubble planetary nebula"],
    "planet":       ["saturn rings hubble", "jupiter hubble", "neptune hubble"],
    "star_cluster": ["hubble globular cluster", "hubble star cluster","omega centauri hubble","hubble open cluster stars"],
}
"""

QUERIES: dict[str, list[str]] = {
    "galaxy":       ["hubble galaxy", "james webb galaxy", "hubble spiral galaxy"],  # already 150, leave it
    "nebula":       ["hubble nebula", "james webb nebula", "hubble planetary nebula", 
                     "crab nebula", "eagle nebula hubble", "orion nebula hubble"],    # add these
    "planet":       ["saturn rings hubble", "jupiter hubble", "neptune hubble",
                     "mars hubble", "hubble planet solar system"],                    # add these
    "star_cluster": ["hubble globular cluster", "hubble star cluster",
                     "omega centauri hubble", "hubble open cluster stars",
                     "pleiades hubble", "47 tucanae hubble"],                         # add these
}


def main():
    client = NASAClient()
    extractor = LabelExtractor()
    downloader = ImageDownloader(
        raw_data_dir  = "data/raw",
        metadata_path = "data/metadata/image_log.csv",
    )

    for target_class, queries in QUERIES.items():
        saved = 0
        logger.info(f"\n{'='*50}")
        logger.info(f"Downloading class: {target_class}")

        for query in queries:
            if saved >= IMAGES_PER_CLASS:
                break

            items = client.search_all_pages(query, max_images=200)
            logger.info(f"  Query '{query}' → {len(items)} items returned")

            for item in items:
                if saved >= IMAGES_PER_CLASS:
                    break

                nasa_id = client.get_nasa_id(item)
                url= client.get_preview_url(item)

                if not nasa_id or not url:
                    continue

                # Only save if the metadata confirms the image matches target class
                extracted = extractor.from_library_item(item)
                if extracted != target_class:
                    continue

                if downloader.save(url, target_class, nasa_id):
                    saved += 1

        logger.info(f"  Saved {saved} images for class '{target_class}'")


if __name__ == "__main__":
    main()
