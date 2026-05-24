import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Keywords that map to each class
LABEL_MAP: dict[str, list[str]] = {
    "galaxy": [
        "galaxy", "galaxies", "milky way", "andromeda",
        "spiral galaxy", "elliptical galaxy", "dwarf galaxy",
    ],
    "nebula": [
        "nebula", "nebulae", "supernova remnant",
        "planetary nebula", "emission nebula", "reflection nebula",
    ],
    "planet": [
        "jupiter", "saturn", "uranus", "neptune", "venus",
        "mercury", "planet", "planetary",
    ],
    "star_cluster": [
        "star cluster", "globular cluster", "open cluster",
        "stellar cluster", "pleiades",
    ],
}

# More-specific labels are checked first to avoid ambiguity
PRIORITY_ORDER = [
    "star_cluster",
    "planet",
    "nebula",
    "galaxy",
]


class LabelExtractor:
    def __init__(self, label_map: dict = None):
        self.label_map = label_map or LABEL_MAP

    def _match(self, text: str) -> Optional[str]:
        text = text.lower()
        for label in PRIORITY_ORDER:
            if any(kw in text for kw in self.label_map[label]):
                return label
        return None

    def from_library_item(self, item: dict) -> Optional[str]:
        """Extract label from NASA Image Library search result."""
        try:
            data    = item.get("data", [{}])[0]
            title   = data.get("title", "")
            desc    = data.get("description", "")
            keywords = " ".join(data.get("keywords", []))
            return self._match(f"{title} {desc} {keywords}")
        except Exception as e:
            logger.warning(f"label extraction failed: {e}")
            return None

    def from_apod(self, apod: dict) -> Optional[str]:
        """Extract label from APOD API response."""
        title       = apod.get("title", "")
        explanation = apod.get("explanation", "")
        return self._match(f"{title} {explanation}")
