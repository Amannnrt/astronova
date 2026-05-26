A production-grade, self-updating machine learning system that automatically fetches NASA astronomy images, classifies them using a fine-tuned ResNet50 model, monitors model confidence for data drift, and triggers automated retraining and redeployment — with zero manual intervention.

---

## Architecture

```
NASA APIs (APOD + Image Library)
            │
            ▼
┌─────────────────────────────────┐
│   Apache Airflow (Astro CLI)    │
│   DAG 1: Fetch APOD daily       │
│   DAG 2: Retrain monthly        │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│   Data Pipeline                 │
│   NASA API → Download → Label   │
│   DVC tracks dataset versions   │
│   DagsHub as remote storage     │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│   Training Pipeline             │
│   ResNet50 fine-tuning          │
│   MLflow tracks experiments     │
│   Model Registry for versioning │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│   Drift Detection               │
│   Confidence score monitoring   │
│   Auto-triggers retraining      │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│   Serving Layer                 │
│   FastAPI inference API         │
│   Gradio frontend demo          │
│   Docker containerized          │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│   CI/CD + Deployment            │
│   GitHub Actions (CI + CD)      │
│   AWS ECR (Docker registry)     │
│   AWS EC2 (production server)   │
└─────────────────────────────────┘
```

---

## Tech Stack

| Layer | Tool | Purpose |
|---|---|---|
| Model | PyTorch + ResNet50 | Fine-tuned CNN classifier |
| Data Source | NASA Image Library + APOD API | Training data + live stream |
| Orchestration | Apache Airflow (Astro CLI) | Scheduling all pipeline stages |
| Data Versioning | DVC + DagsHub | Track dataset versions across runs |
| Experiment Tracking | MLflow + Model Registry | Log metrics, params, and models |
| CI/CD | GitHub Actions | Automated test, build, deploy |
| Containerization | Docker | Reproducible environments |
| Serving | FastAPI | Model inference API |
| Frontend | Gradio | Interactive demo UI |
| Deployment | AWS EC2 + ECR | Cloud hosting and image registry |

---

## Features

**Self-updating data pipeline** — Airflow fetches a new NASA APOD image every day automatically and appends it to the dataset.

**Automated retraining** — Every month, Airflow triggers a full retraining pipeline via DVC. The new model is evaluated against the production model in MLflow Registry and promoted only if it improves.

**Drift detection** — Every prediction is logged to a rolling confidence window. When average confidence drops below 0.75, the drift detector flags it and triggers the retraining DAG automatically.

**Reproducible pipelines** — DVC tracks every version of the dataset. Every training run is tied to a specific data snapshot so results are fully reproducible with `dvc repro`.

**CI/CD** — Every push to main runs linting and unit tests. On success, the Docker image is built, pushed to AWS ECR, and deployed to EC2 automatically.

---

## Model

| Detail | Value |
|---|---|
| Architecture | ResNet50 (pretrained on ImageNet) |
| Fine-tuned layers | Layer4 + FC head |
| Classes | Galaxy, Nebula, Planet, Star Cluster |
| Training images | ~400 (NASA Image Library) |
| Test accuracy | 77.05% |
| Weighted F1 | 0.772 |
| Avg confidence | 0.836 |
| Optimizer | AdamW |
| Scheduler | CosineAnnealingLR |
| Dropout | 0.5 |

---

## Project Structure

```
astronova/
├── .github/
│   └── workflows/
│       ├── ci.yml              # Lint + test on every push
│       └── cd.yml              # Build, push to ECR, deploy to EC2
├── airflow/
│   └── dags/
│       ├── fetch_apod_dag.py   # Daily APOD fetch
│       └── retrain_dag.py      # Monthly retraining pipeline
├── configs/
│   ├── data_config.yaml
│   ├── model_config.yaml
│   └── training_config.yaml
├── data/
│   ├── raw/                    # DVC tracked
│   ├── processed/              # DVC tracked
│   └── metadata/
│       ├── image_log.csv       # Every fetched image logged
│       └── drift_log.json      # Drift events logged
├── docker/
│   └── Dockerfile
├── scripts/
│   └── seed_data.py            # One-time historical data download
├── src/
│   ├── data/
│   │   ├── nasa_client.py      # NASA API wrapper
│   │   ├── downloader.py       # Image download + metadata logging
│   │   ├── label_extractor.py  # Extract class labels from NASA metadata
│   │   └── preprocessor.py    # Resize, normalize, augment, split
│   ├── models/
│   │   └── resnet_classifier.py # ResNet50 with custom classification head
│   ├── monitoring/
│   │   └── drift_detector.py   # Confidence monitoring + auto-trigger
│   ├── serving/
│   │   ├── fastapi_app.py      # Inference API
│   │   └── gradio_app.py       # Frontend demo
│   └── training/
│       ├── train.py            # Training loop with MLflow logging
│       └── evaluate.py         # Test set evaluation
├── tests/
│   └── unit/
│       └── test_model.py
├── dvc.yaml                    # Pipeline stage definitions
├── params.yaml                 # Pipeline parameters
└── requirements.txt
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Docker
- Astro CLI (for Airflow)
- NASA API key — free at https://api.nasa.gov

### Installation

```bash
# Clone the repository
git clone https://github.com/iammdaman2004/astronova.git
cd astronova

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install CPU-only torch first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file in the project root:

```env
NASA_API_KEY=your_nasa_api_key
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
CONFIDENCE_THRESHOLD=0.75
RETRAIN_TRIGGER_COUNT=300
AIRFLOW_URL=http://localhost:8080
AIRFLOW_USERNAME=admin
AIRFLOW_PASSWORD=admin
```

---

## Running the Project

### 1. Download Training Data

```bash
python scripts/seed_data.py
```

### 2. Train the Model

```bash
python -m src.training.train
```

### 3. Evaluate on Test Set

```bash
python -m src.training.evaluate
```

### 4. View MLflow Dashboard

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open `http://localhost:5000`

### 5. Run the Full DVC Pipeline

```bash
dvc repro
```

### 6. Start the API

```bash
uvicorn src.serving.fastapi_app:app --reload --port 8000
```

### 7. Start the Gradio Demo

```bash
python -m src.serving.gradio_app
```

Open `http://localhost:7860`

### 8. Start Airflow

```bash
cd airflow
astro dev start
```

Open `http://airflow.localhost:6563`

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Server status, uptime, model info |
| `/predict` | POST | Classify an astronomy image |
| `/metrics` | GET | Prediction count, avg confidence, drift flag |
| `/model-info` | GET | Model architecture, accuracy, classes |

### Example prediction request

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@galaxy.jpg"
```

### Example response

```json
{
  "class": "galaxy",
  "confidence": 0.8912,
  "all_scores": {
    "galaxy": 0.8912,
    "nebula": 0.0621,
    "planet": 0.0312,
    "star_cluster": 0.0155
  },
  "inference_time_ms": 87.14,
  "prediction_number": 42
}
```

---

## Drift Detection

Every prediction logs its softmax confidence score to a rolling window of 100 predictions. When the average drops below the threshold (default 0.75), the drift detector:

1. Logs the drift event to `data/metadata/drift_log.json`
2. Triggers the `retrain_monthly` Airflow DAG via REST API
3. Sets a flag to prevent duplicate triggers until retraining completes

The drift flag is visible at `GET /metrics`:

```json
{
  "total_predictions": 150,
  "avg_confidence": 0.71,
  "is_drifting": true,
  "threshold": 0.75,
  "drift_triggered": true
}
```

---

## CI/CD Pipeline

**CI** runs on every push to main:
- Python linting with flake8
- Unit tests with pytest

**CD** runs on every successful push to main:
- Builds Docker image
- Pushes to AWS ECR
- SSHs into EC2 and pulls + restarts the container

```
Push to main
     │
     ▼
CI: lint + test
     │
     ▼
CD: docker build → push to ECR → deploy to EC2
     │
     ▼
FastAPI live at http://EC2_IP:8000
```

---

## Retraining Pipeline

The monthly retraining DAG runs 4 tasks in sequence:

```
update_data_version
        │  dvc add + dvc push → DagsHub
        ▼
run_training_pipeline
        │  dvc repro → preprocess + train
        ▼
evaluate_new_model
        │  test set metrics logged to MLflow
        ▼
compare_models
        │  new model vs production model
        ├── better → promote to production
        └── worse  → archive, keep current
```

---

## Results

| Metric | Value |
|---|---|
| Test Accuracy | 77.05% |
| Weighted F1 | 0.772 |
| F1 — Galaxy | 0.769 |
| F1 — Nebula | 0.667 |
| F1 — Planet | 0.919 |
| F1 — Star Cluster | 0.636 |
| Avg Inference Time | ~87ms |
| Avg Confidence | 0.836 |

---

## Future Improvements

- Expand dataset with more classes (solar events, deep field images)
- Add semantic search using CLIP embeddings and FAISS
- Deploy Gradio demo to HuggingFace Spaces for public access
- Add Prometheus + Grafana for production monitoring
- Implement A/B testing for model promotion decisions
- Add data quality checks before retraining triggers

---

## License

MIT License — see LICENSE file for details.
