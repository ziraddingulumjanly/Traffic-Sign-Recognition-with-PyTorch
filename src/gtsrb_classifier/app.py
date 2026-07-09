"""FastAPI application for serving the GTSRB traffic sign classifier.

Author: Ziraddin Gulumjanli, 2026

Run locally:
    uvicorn gtsrb_classifier.app:app --host 0.0.0.0 --port 8000

The API expects MODEL_PATH to point to a trained checkpoint. If no checkpoint is
available, the health endpoint clearly reports that the model is not loaded.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile

from gtsrb_classifier.predict import load_trained_model, predict_image_bytes
from gtsrb_classifier.utils import get_device

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/gtsrb_resnet18_best.pt"))
TOP_K = int(os.getenv("TOP_K", "5"))

app = FastAPI(
    title="GTSRB Traffic Sign Classifier",
    description="Production-style PyTorch inference API for German traffic sign recognition.",
    version="1.0.0",
)

DEVICE: torch.device = get_device()
MODEL: torch.nn.Module | None = None
IMAGE_SIZE: int = 64


@app.on_event("startup")
def load_model_on_startup() -> None:
    """Load the trained model once when the API starts."""
    global MODEL, IMAGE_SIZE
    if not MODEL_PATH.exists():
        MODEL = None
        return

    MODEL, metadata = load_trained_model(MODEL_PATH, device=DEVICE)
    config = metadata.get("config", {})
    IMAGE_SIZE = int(config.get("image_size", IMAGE_SIZE))


@app.get("/health")
def health() -> dict[str, Any]:
    """Return service and model-loading status."""
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model_path": str(MODEL_PATH),
        "model_loaded": MODEL is not None,
    }


@app.post("/predict")
async def predict(file: Annotated[UploadFile, File(...)]) -> dict[str, Any]:
    """Predict the traffic sign class from an uploaded image."""
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model checkpoint not found at {MODEL_PATH}. Train a model or set MODEL_PATH.",
        )
    if file.content_type is not None and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    image_bytes = await file.read()
    predictions = predict_image_bytes(
        image_bytes=image_bytes,
        model=MODEL,
        image_size=IMAGE_SIZE,
        top_k=TOP_K,
        device=DEVICE,
    )
    return {"filename": file.filename, "predictions": predictions}
