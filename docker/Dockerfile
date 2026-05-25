# docker/Dockerfile

FROM python:3.11-slim

WORKDIR /app

# Install CPU-only torch first — avoids downloading 2GB of CUDA
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY configs/ ./configs/

EXPOSE 8000

CMD ["uvicorn", "src.serving.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
