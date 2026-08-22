"""
main.py
-------
FastAPI service exposing the trained defect-classification model as a
REST API, mirroring how this model would be wired into a real production
inspection line (camera -> API call -> ACCEPT/REJECT decision).

Run:
    uvicorn api.main:app --reload --port 8000

Docs:
    http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Allow importing from src/ regardless of current working directory.
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from inference import InferenceEngine  # noqa: E402
from schemas import ErrorResponse, HealthResponse, PredictionResponse  # noqa: E402

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "models" / "best_model.pth"
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/bmp"}

engine: InferenceEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    if CHECKPOINT_PATH.exists():
        engine = InferenceEngine(CHECKPOINT_PATH)
        print(f"Loaded model from {CHECKPOINT_PATH} | classes={engine.classes}")
    else:
        engine = None
        print(
            f"WARNING: no checkpoint found at {CHECKPOINT_PATH}. "
            "Train the model first (python src/train.py) — /predict will 503 until then."
        )
    yield


app = FastAPI(
    title="Manufacturing Quality Inspection API",
    description="CNN-based defect classification service for production-line quality inspection.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if engine is not None else "model_not_loaded",
        model_loaded=engine is not None,
        classes=engine.classes if engine is not None else [],
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={503: {"model": ErrorResponse}, 400: {"model": ErrorResponse}},
)
async def predict(file: UploadFile = File(...)):
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Train it first with `python src/train.py`.",
        )

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type '{file.content_type}'. Allowed: {sorted(ALLOWED_CONTENT_TYPES)}",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = engine.predict(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return PredictionResponse(**result)


@app.get("/")
def root():
    return {
        "service": "Manufacturing Quality Inspection API",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict (multipart/form-data, field name 'file')",
    }
