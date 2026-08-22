"""
test_api.py
-----------
Integration tests for the FastAPI service using TestClient (no live server
required). A small untrained checkpoint is created on the fly in a temp
directory so these tests never depend on the real trained model artifact
or the real dataset — they verify the API *wiring*, not model accuracy.
"""

import io
import sys
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
sys.path.append(str(Path(__file__).resolve().parent.parent / "api"))

from model import build_model  # noqa: E402


@pytest.fixture(scope="module")
def temp_checkpoint(tmp_path_factory):
    """Create a throwaway (untrained) checkpoint so the API has something to load."""
    models_dir = tmp_path_factory.mktemp("models")
    classes = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
    model = build_model(num_classes=len(classes))
    ckpt_path = models_dir / "best_model.pth"
    torch.save({"model_state_dict": model.state_dict(), "classes": classes, "val_acc": 0.0, "epoch": 0}, ckpt_path)
    return ckpt_path


@pytest.fixture()
def client(temp_checkpoint, monkeypatch):
    # Point the API at our throwaway checkpoint before importing/instantiating it.
    import api.main as main_module

    monkeypatch.setattr(main_module, "CHECKPOINT_PATH", temp_checkpoint)
    with TestClient(main_module.app) as c:
        yield c


def _dummy_image_bytes(fmt="JPEG") -> bytes:
    img = Image.new("RGB", (224, 224), color=(120, 120, 120))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def test_root_endpoint(client):
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert "service" in body


def test_health_endpoint_reports_model_loaded(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_loaded"] is True
    assert len(body["classes"]) == 6


def test_predict_with_valid_image_returns_expected_schema(client):
    img_bytes = _dummy_image_bytes()
    resp = client.post("/predict", files={"file": ("test.jpg", img_bytes, "image/jpeg")})
    assert resp.status_code == 200
    body = resp.json()

    assert "predicted_class" in body
    assert "confidence" in body
    assert "inspection_status" in body
    assert "class_probabilities" in body
    assert body["inspection_status"] in {"ACCEPT", "REJECT", "MANUAL_REVIEW"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert abs(sum(body["class_probabilities"].values()) - 1.0) < 1e-3


def test_predict_rejects_unsupported_content_type(client):
    resp = client.post("/predict", files={"file": ("test.txt", b"not an image", "text/plain")})
    assert resp.status_code == 400


def test_predict_rejects_empty_file(client):
    resp = client.post("/predict", files={"file": ("empty.jpg", b"", "image/jpeg")})
    assert resp.status_code == 400
