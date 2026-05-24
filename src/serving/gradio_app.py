"""
Gradio frontend demo.
Run locally: python src/serving/gradio_app.py
Make sure FastAPI is running on port 8000 first.
"""
import io
import os
from dotenv import load_dotenv
import gradio as gr
import requests
from PIL import Image

from src.data.nasa_client import NASAClient
load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")


def predict_image(image: Image.Image):
    if image is None:
        return {}, "No image provided."

    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)

    try:
        response = requests.post(
            f"{API_URL}/predict",
            files={"file": ("image.jpg", buf, "image/jpeg")},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        pred_class = data["class"].replace("_", " ").title()
        confidence = data["confidence"]
        all_scores = data["all_scores"]
        inf_time   = data["inference_time_ms"]

        info = f"Predicted: {pred_class}  |  Confidence: {confidence:.1%}  |  Inference: {inf_time}ms"
        return all_scores, info

    except requests.exceptions.ConnectionError:
        return {}, "Cannot connect to API. Make sure FastAPI is running on port 8000."
    except Exception as e:
        return {}, f"Error: {str(e)}"

def fetch_apod():
    try:
        client = NASAClient()
        # Try today first, fall back to a few previous dates
        from datetime import datetime, timedelta
        for days_back in range(0, 10):
            date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            apod = client.get_apod(date=date)
            print(f"Trying {date}: {apod}")
            if apod and apod.get("url"):
                response = requests.get(apod["url"], timeout=30)
                return Image.open(io.BytesIO(response.content)).convert("RGB")
        print("No image found in last 10 days")
    except Exception as e:
        print(f"APOD fetch failed: {e}")
    return None

with gr.Blocks(
    theme=gr.themes.Monochrome(),
    title="AstroNova",
    css=".gradio-container { max-width: 900px; margin: auto; }"
) as demo:

    gr.Markdown("## AstroNova — Astronomy Image Classifier")
    gr.Markdown("Upload an astronomy image or pull today's NASA image to classify it.")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="Input Image", type="pil", height=300)
            with gr.Row():
                fetch_btn   = gr.Button("Fetch NASA APOD", variant="secondary")
                predict_btn = gr.Button("Classify", variant="primary")

        with gr.Column(scale=1):
            label_output = gr.Label(label="Class Scores", num_top_classes=4)
            result_text  = gr.Textbox(label="Result", lines=2)

    gr.Markdown(
        "Classes: Galaxy | Nebula | Planet | Star Cluster  —  "
        "ResNet50 | Test Accuracy: 77.05% | F1: 0.772"
    )

    predict_btn.click(predict_image, inputs=[image_input], outputs=[label_output, result_text])
    fetch_btn.click(fetch_apod, inputs=[], outputs=[image_input])


if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)
