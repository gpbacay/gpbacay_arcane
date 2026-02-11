"""
FastAPI server for ARCANE MNIST inference.
Serves the trained model so the docs site (or any client) can get predictions on drawn digits.

Run from repo root:
  python examples/serve_mnist_api.py

Or with custom host/port:
  MNIST_WEIGHTS_PATH=path/to/weights.h5 uvicorn examples.serve_mnist_api:app --host 0.0.0.0 --port 8000
"""
import os
import sys
import base64
import io
import numpy as np

# Repo root = parent of examples/
EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(EXAMPLES_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Optional: avoid TF logs if desired
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import model builder from the same repo
sys.path.insert(0, EXAMPLES_DIR)
from mnist_arcane import build_mnist_arcane_model

app = FastAPI(title="ARCANE MNIST Inference", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model and weights path
WEIGHTS_PATH = os.environ.get(
    "MNIST_WEIGHTS_PATH",
    os.path.join(ROOT_DIR, "mnist_arcane_model.weights.h5"),
)
model = None


class PredictRequest(BaseModel):
    """Base64-encoded image (PNG/JPEG or raw 28x28 grayscale)."""
    image_base64: str


class PredictResponse(BaseModel):
    digit: int
    confidence: float
    probabilities: list[float]


def preprocess_for_model(image_bytes: bytes) -> np.ndarray:
    """
    Decode image, resize to 28x28, grayscale, 0–255 range.
    Model applies Rescaling(1/255) internally; we pass raw 0–255.
    """
    try:
        from PIL import Image
    except ImportError:
        raise HTTPException(status_code=500, detail="PIL required for image decode. pip install Pillow")
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    data = np.array(img).astype(np.float32)
    if np.mean(data) > 127.5:
        data = 255.0 - data
    return np.expand_dims(data, axis=0)


@app.on_event("startup")
def load_model():
    global model
    if not os.path.exists(WEIGHTS_PATH):
        raise RuntimeError(
            f"Weights not found: {WEIGHTS_PATH}. "
            "Train with: python examples/mnist_arcane.py"
        )
    model = build_mnist_arcane_model(
        persistent_predictive=True,
        bioplastic_inference_plasticity=True,
    )
    model(np.zeros((1, 28, 28), dtype=np.float32))
    model.load_weights(WEIGHTS_PATH)
    print(f"Loaded weights from {WEIGHTS_PATH}")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        raw = base64.b64decode(req.image_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64: {e}")
    try:
        batch = preprocess_for_model(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decode failed: {e}")
    probs = model.predict(batch, verbose=0)
    prob_list = probs[0].tolist()
    digit = int(np.argmax(prob_list))
    confidence = float(prob_list[digit])
    return PredictResponse(digit=digit, confidence=confidence, probabilities=prob_list)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
