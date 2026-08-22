"""
model.py
--------
Transfer-learning model definition: ImageNet-pretrained ResNet18 with a
replaced classification head for the N-class defect problem.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    """Build a ResNet18 classifier.

    Args:
        num_classes: number of defect categories.
        freeze_backbone: if True, freeze all convolutional layers and only
            train the new fully-connected head (fast, good for small
            datasets like NEU-CLS and avoids overfitting on ~1,800 images).
            Set False to fine-tune the whole network once the head has
            converged (optional second training phase).
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    # New head is always trainable, even if backbone is frozen.
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def unfreeze_backbone(model: nn.Module) -> nn.Module:
    """Unfreeze all layers for a fine-tuning phase (use a low LR after this)."""
    for param in model.parameters():
        param.requires_grad = True
    return model


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = build_model(num_classes=6)
    dummy = torch.randn(2, 3, 224, 224)
    out = m(dummy)
    print(f"Output shape: {out.shape}")  # expect (2, 6)
    print(f"Trainable params: {count_trainable_params(m):,}")
