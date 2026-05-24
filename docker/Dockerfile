# docker/Dockerfile

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Install CPU-only torch FIRST — avoids downloading 2GB of CUDA
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Then install everything else
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "src.serving.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
