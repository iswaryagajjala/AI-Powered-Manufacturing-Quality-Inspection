"""
inference.py
------------
Single-image inference pipeline shared by the CLI and the FastAPI service.

Loads a trained checkpoint once (InferenceEngine) and exposes a simple
`predict(image_bytes_or_path)` -> dict interface.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
from PIL import Image

from model import build_model
from preprocessing import get_eval_transforms

# Classes considered "no defect" -> ACCEPT. NEU-CLS has no such class, so
# this is empty by default; adapt it if you plug in a dataset that has a
# "good"/"no_defect" label.
NO_DEFECT_CLASSES: set[str] = set()

# Below this confidence, flag for manual human review rather than trusting
# either an ACCEPT or REJECT prediction outright.
LOW_CONFIDENCE_THRESHOLD = 0.60


class InferenceEngine:
    def __init__(self, checkpoint_path: str | Path, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.classes: list[str] = ckpt["classes"]

        self.model = build_model(num_classes=len(self.classes))
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.transform = get_eval_transforms()

    def _load_image(self, image: Union[str, Path, bytes]) -> Image.Image:
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Could not read image at {image} (corrupted or unsupported format)")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return Image.fromarray(img_rgb)
        elif isinstance(image, bytes):
            arr = np.frombuffer(image, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode image bytes (corrupted or unsupported format)")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return Image.fromarray(img_rgb)
        else:
            raise TypeError(f"Unsupported image input type: {type(image)}")

    @torch.no_grad()
    def predict(self, image: Union[str, Path, bytes]) -> dict:
        pil_image = self._load_image(image)
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        outputs = self.model(tensor)
        probs = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()

        pred_idx = int(np.argmax(probs))
        pred_class = self.classes[pred_idx]
        confidence = float(probs[pred_idx])

        if confidence < LOW_CONFIDENCE_THRESHOLD:
            status = "MANUAL_REVIEW"
        elif pred_class in NO_DEFECT_CLASSES:
            status = "ACCEPT"
        else:
            status = "REJECT"

        return {
            "predicted_class": pred_class,
            "confidence": round(confidence, 4),
            "inspection_status": status,
            "class_probabilities": {
                cls: round(float(p), 4) for cls, p in zip(self.classes, probs)
            },
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a single image")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--checkpoint", default="models/best_model.pth")
    args = parser.parse_args()

    engine = InferenceEngine(args.checkpoint)
    result = engine.predict(args.image)

    print("Prediction result:")
    for k, v in result.items():
        print(f"  {k}: {v}")
