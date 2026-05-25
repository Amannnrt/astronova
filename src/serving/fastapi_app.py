"""
FastAPI serving app.
Run locally: uvicorn src.serving.fastapi_app:app --reload --port 8000
"""
import io
import logging
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Optional
import os

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms

from src.models.resnet_classifier import AstroClassifier

from src.monitoring.drift_detector import DriftDetector
drift_detector = DriftDetector(
    window_size=100,
    threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.75")),
)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────
CLASS_NAMES   = ["galaxy", "nebula", "planet", "star_cluster"]
MODEL_PATH    = "models/best_model.pt"
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EVAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# App state 
model= None
prediction_count = 0
confidence_window = deque(maxlen=100)   # rolling window for drift detection
start_time       = time.time()


#Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    logger.info(f"Loading model from {MODEL_PATH} on {DEVICE}")
    model = AstroClassifier.load(MODEL_PATH, device=str(DEVICE))
    model.to(DEVICE)
    model.eval()
    logger.info("Model loaded successfully")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="AstroNova API",
    description="NASA Astronomy Image Classifier",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess(image_bytes: bytes) -> torch.Tensor:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return EVAL_TRANSFORMS(image).unsqueeze(0).to(DEVICE)


@app.get("/health")
def health():
    uptime_seconds = int(time.time() - start_time)
    hours, rem     = divmod(uptime_seconds, 3600)
    minutes, secs  = divmod(rem, 60)
    return {
        "status":       "ok",
        "model":        MODEL_PATH,
        "device":       str(DEVICE),
        "uptime":       f"{hours}h {minutes}m {secs}s",
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global prediction_count

    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    image_bytes = await file.read()

    try:
        tensor = preprocess(image_bytes)
    except Exception as e:
        raise HTTPException(400, f"Could not process image: {e}")

    start = time.time()
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze()

    inference_ms = round((time.time() - start) * 1000, 2)

    confidence, pred_idx = probs.max(dim=0)
    confidence           = float(confidence)
    pred_class           = CLASS_NAMES[pred_idx.item()]

    prediction_count += 1
    drift_detector.log_confidence(confidence, pred_class)
    drift_detector.check_and_trigger()
    
    return {
        "class":            pred_class,
        "confidence":       round(confidence, 4),
        "all_scores":       {
            cls: round(float(probs[i]), 4)
            for i, cls in enumerate(CLASS_NAMES)
        },
        "inference_time_ms": inference_ms,
        "prediction_number": prediction_count,
    }


@app.get("/metrics")
def metrics():
    status = drift_detector.get_status()
    return {
        "total_predictions":  prediction_count,
        **status,

        }


@app.get("/model-info")
def model_info():
    return {
        "architecture":    "ResNet50",
        "num_classes":     len(CLASS_NAMES),
        "class_names":     CLASS_NAMES,
        "model_path":      MODEL_PATH,
        "test_accuracy":   0.7705,
        "test_f1":         0.7722,
        "avg_confidence":  0.8363,
    }
