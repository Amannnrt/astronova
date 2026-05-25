import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)

CLASS_NAMES = [
    "galaxy", "nebula", "planet", "star_cluster",
    ]


class AstroClassifier(nn.Module):
    def __init__(self, num_classes: int = 4, dropout: float = 0.5):
        super().__init__()

        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Freeze everything up to (but not including) layer4
        for name, param in backbone.named_parameters():
            param.requires_grad = "layer4" in name or  "fc" in name

        # Replace the final fully-connected layer with a custom head
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

        self.model = backbone
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @property
    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    #Persistence
    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "num_classes":      self.num_classes,
                "class_names":      CLASS_NAMES,
            },
            path,
        )
        logger.info(f"Model saved → {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "AstroClassifier":
        ckpt  = torch.load(path, map_location=device)
        model = cls(num_classes=ckpt["num_classes"])
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model
