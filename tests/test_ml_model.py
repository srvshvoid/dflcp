"""Unit tests for FedForge ML model architecture and state serialization (src/fedforge/ml/model.py).

Tests:
1. Model forward pass and 1-logit output shape.
2. Backbone freezing (only classification head is trainable on CPU).
3. Round-trip state identity (set_state(get_state(m)) preserves all values).
4. State-to-vector and vector-to-state serialization accuracy.
5. BatchNorm buffers (running_mean, running_var, num_batches_tracked) included in full state.
6. Verification of head-only vector size (513 parameters).
"""

from __future__ import annotations

import unittest
import torch
import torch.nn as nn

from fedforge.ml.model import (
    FEDERATION_CONTRACT,
    ResNet18Binary,
    SmallCNN,
    build_model,
    get_state,
    get_state_schema,
    set_state,
    state_to_vector,
    vector_to_state,
)


class TestMLModel(unittest.TestCase):
    """Test suite for neural network architectures and state serialization."""

    def test_build_model_forward_shape(self) -> None:
        """Verify model accepts (B, 3, H, W) and outputs a single binary logit (B, 1)."""
        model = build_model(model_type="resnet18", pretrained=False, freeze_backbone=True)
        model.eval()

        dummy_input = torch.randn(4, 3, 64, 64)
        with torch.no_grad():
            output = model(dummy_input)

        self.assertEqual(output.shape, (4, 1))
        self.assertEqual(output.dtype, torch.float32)

    def test_backbone_freezing(self) -> None:
        """Verify freeze_backbone=True freezes all feature layers except fc head."""
        frozen_model = build_model(model_type="resnet18", pretrained=False, freeze_backbone=True)
        trainable_params = [name for name, p in frozen_model.named_parameters() if p.requires_grad]
        frozen_params = [name for name, p in frozen_model.named_parameters() if not p.requires_grad]

        # Only fc.weight and fc.bias must be trainable
        self.assertEqual(sorted(trainable_params), ["backbone.fc.bias", "backbone.fc.weight"])
        self.assertGreater(len(frozen_params), 50)

        # Unfrozen model
        unfrozen_model = build_model(model_type="resnet18", pretrained=False, freeze_backbone=False)
        all_trainable = [name for name, p in unfrozen_model.named_parameters() if p.requires_grad]
        self.assertEqual(len(all_trainable), len(list(unfrozen_model.parameters())))

    def test_smallcnn_fallback(self) -> None:
        """Verify SmallCNN fallback architecture."""
        small = build_model(model_type="smallcnn")
        dummy = torch.randn(2, 3, 32, 32)
        out = small(dummy)
        self.assertEqual(out.shape, (2, 1))

    def test_get_state_and_head_vector_size(self) -> None:
        """Verify get_state returns 513 parameters in head scope."""
        model = build_model(model_type="resnet18", pretrained=False, freeze_backbone=True)
        head_state = get_state(model, scope="head")

        self.assertIn("backbone.fc.weight", head_state)
        self.assertIn("backbone.fc.bias", head_state)

        flat_vector = state_to_vector(head_state)
        self.assertEqual(len(flat_vector), FEDERATION_CONTRACT["head_parameter_count"])
        self.assertEqual(len(flat_vector), 513)

    def test_round_trip_state_identity(self) -> None:
        """Verify set_state(get_state(m)) is an exact identity."""
        model = build_model(model_type="resnet18", pretrained=False)
        original_state = get_state(model, scope="full")

        # Mutate model weights with noise
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p))

        # Restore state
        set_state(model, original_state, strict=True)
        restored_state = get_state(model, scope="full")

        for key in original_state:
            self.assertTrue(
                torch.equal(original_state[key], restored_state[key]),
                f"Mismatch in key {key} after set_state restoration!",
            )

    def test_batchnorm_buffers_included_in_full_state(self) -> None:
        """Verify BatchNorm running_mean, running_var, and num_batches_tracked are captured."""
        model = build_model(model_type="resnet18", pretrained=False)
        full_state = get_state(model, scope="full")

        # Check for specific BatchNorm buffers
        bn_mean_keys = [k for k in full_state if "running_mean" in k]
        bn_var_keys = [k for k in full_state if "running_var" in k]
        bn_batches_keys = [k for k in full_state if "num_batches_tracked" in k]

        self.assertGreater(len(bn_mean_keys), 10)
        self.assertGreater(len(bn_var_keys), 10)
        self.assertGreater(len(bn_batches_keys), 10)

        # Mutate a buffer and verify set_state loads it
        sample_key = bn_mean_keys[0]
        test_value = torch.ones_like(full_state[sample_key]) * 99.0
        full_state[sample_key] = test_value

        set_state(model, full_state, strict=True)
        reloaded = get_state(model, scope="full")
        self.assertTrue(torch.equal(reloaded[sample_key], test_value))

    def test_vector_serialization_round_trip(self) -> None:
        """Verify state_to_vector -> vector_to_state preserves exact tensor values."""
        model = build_model(model_type="resnet18", pretrained=False)
        head_state = get_state(model, scope="head")
        schema = get_state_schema(head_state)

        vector = state_to_vector(head_state)
        reconstructed = vector_to_state(vector, schema)

        self.assertEqual(set(head_state.keys()), set(reconstructed.keys()))
        for k in head_state:
            self.assertTrue(
                torch.equal(head_state[k], reconstructed[k]),
                f"Tensor {k} mismatch after vector reconstruction!",
            )


if __name__ == "__main__":
    unittest.main()
