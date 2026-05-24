import logging
import shutil 
from pathlib import Path
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CLASS_NAMES = [
    "galaxy", "nebula", "planet", "star_cluster",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

IMAGE_SIZE = 224


logger = logging.getLogger(__name__)

#Raw JPG -> resize->crop->flip/rotate->convert to tensoor->normalize pixel values


TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

EVAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class AstroDataset(Dataset):
    def __init__(self, paths: list[str], labels: list[int], transform=None):
        self.paths     = paths
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.paths[idx]).convert("RGB")
        except Exception:
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))

        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


class DataPreprocessor:
    def __init__(
        self,
        raw_dir: str,
        processed_dir: str,
        train_split: float = 0.70,
        val_split:   float = 0.15,
    ):
        self.raw_dir       = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.train_split   = train_split
        self.val_split     = val_split

    def _collect(self) -> tuple[list[str], list[int]]:
        paths, labels = [], []
        for cls in CLASS_NAMES:
            cls_dir = self.raw_dir / cls
            if not cls_dir.exists():
                continue
            for p in cls_dir.glob("*.[jp][pn]g"):
                paths.append(str(p))
                labels.append(CLASS_TO_IDX[cls])
        return paths, labels

    def split_and_copy(self):
        """Split raw images into train/val/test and copy to processed dir."""
        paths, labels = self._collect()

        if not paths:
            raise FileNotFoundError(f"No images found in {self.raw_dir}. Run seed_data.py first.")

        logger.info(f"Total images: {len(paths)}")

        # First split: train vs rest
        test_size = 1 - self.train_split
        tr_paths, tmp_paths, tr_labels, tmp_labels = train_test_split(
            paths, labels, test_size=test_size, random_state=42, stratify=labels
        )

        # Second split: val vs test
        val_ratio = self.val_split / (self.val_split + (1 - self.train_split - self.val_split))
        va_paths, te_paths, va_labels, te_labels = train_test_split(
            tmp_paths, tmp_labels, test_size=(1 - val_ratio), random_state=42, stratify=tmp_labels
        )

        for split, ps, ls in [("train", tr_paths, tr_labels),
                               ("val",   va_paths, va_labels),
                               ("test",  te_paths, te_labels)]:
            for path, label in zip(ps, ls):
                dest = self.processed_dir / split / CLASS_NAMES[label]
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest / Path(path).name)

        logger.info(f"Train {len(tr_paths)} | Val {len(va_paths)} | Test {len(te_paths)}")


    def _load_split(self, split: str) -> tuple[list[str], list[int]]:
        paths, labels = [], []
        for cls in CLASS_NAMES:
            cls_dir = self.processed_dir / split / cls
            if not cls_dir.exists():
                continue
            for p in cls_dir.glob("*.[jp][pn]g"):
                paths.append(str(p))
                labels.append(CLASS_TO_IDX[cls])
        return paths, labels

    def get_dataloaders(self, batch_size: int = 32) -> tuple[DataLoader, DataLoader, DataLoader]:
        tr = self._load_split("train")
        va = self._load_split("val")
        te = self._load_split("test")

        train_ds = AstroDataset(*tr, TRAIN_TRANSFORMS)
        val_ds   = AstroDataset(*va, EVAL_TRANSFORMS)
        test_ds  = AstroDataset(*te, EVAL_TRANSFORMS)

        kwargs = dict(batch_size=batch_size, num_workers=2, pin_memory=True)
        return (
            DataLoader(train_ds, shuffle=True,  **kwargs),
            DataLoader(val_ds,   shuffle=False, **kwargs),
            DataLoader(test_ds,  shuffle=False, **kwargs),
        )
