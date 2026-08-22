"""
test_model.py
-------------
Unit tests for model construction and forward pass shape/behavior.
Does not require the trained checkpoint or dataset — pure architecture tests.
"""

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from model import build_model, count_trainable_params, unfreeze_backbone  # noqa: E402


def test_model_output_shape():
    num_classes = 6
    model = build_model(num_classes=num_classes)
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (4, num_classes)


def test_model_output_is_logits_not_probabilities():
    model = build_model(num_classes=6)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    # Raw logits should not already sum to 1 per row (that only happens after softmax)
    row_sums = output.sum(dim=1)
    assert not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)


def test_frozen_backbone_has_fewer_trainable_params_than_unfrozen():
    frozen_model = build_model(num_classes=6, freeze_backbone=True)
    frozen_params = count_trainable_params(frozen_model)

    unfrozen_model = unfreeze_backbone(build_model(num_classes=6, freeze_backbone=True))
    unfrozen_params = count_trainable_params(unfrozen_model)

    assert frozen_params < unfrozen_params


def test_model_handles_variable_batch_size():
    model = build_model(num_classes=6)
    for batch_size in (1, 3, 8):
        dummy_input = torch.randn(batch_size, 3, 224, 224)
        output = model(dummy_input)
        assert output.shape[0] == batch_size


def test_different_num_classes():
    for n in (2, 6, 10):
        model = build_model(num_classes=n)
        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)
        assert output.shape == (2, n)
