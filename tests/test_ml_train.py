"""Unit tests for FedForge centralized baseline and sanity checks (src/fedforge/ml/train.py).

Tests:
1. compute_metrics correctness on hand-crafted binary predictions.
2. Threshold selection on validation data.
3. Automated sanity check A: Overfitting 32 samples to ~100% accuracy.
4. Centralized baseline training run with real artifact generation.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from fedforge.ml.data import create_synthetic_image_fixture, load_image_dataset, split_dataset
from fedforge.ml.train import (
    compute_metrics,
    sanity_check_overfit_32,
    select_best_threshold,
    train_centralized,
)


class TestMLTrain(unittest.TestCase):
    """Test suite for centralized baseline trainer and sanity check procedures."""

    def test_compute_metrics_accuracy(self) -> None:
        """Verify metric calculation with known binary vectors."""
        y_true = [0, 0, 1, 1]
        y_prob = [0.1, 0.4, 0.8, 0.9]
        m = compute_metrics(y_true, y_prob, threshold=0.5)

        self.assertEqual(m["accuracy"], 1.0)
        self.assertEqual(m["precision"], 1.0)
        self.assertEqual(m["recall"], 1.0)
        self.assertEqual(m["f1"], 1.0)
        self.assertEqual(m["roc_auc"], 1.0)
        self.assertEqual(m["confusion_matrix"], [[2, 0], [0, 2]])

    def test_select_best_threshold(self) -> None:
        """Verify threshold optimization selects the threshold that maximizes F1."""
        y_true = [0, 0, 1, 1]
        # At threshold 0.5, sample 1 is false positive (0.6 >= 0.5)
        # At threshold 0.65, sample 1 is true negative (0.6 < 0.65)
        y_prob = [0.2, 0.6, 0.7, 0.9]
        thresh, best_f1 = select_best_threshold(y_true, y_prob, thresholds=[0.5, 0.65])
        self.assertEqual(thresh, 0.65)
        self.assertEqual(best_f1, 1.0)

    def test_sanity_check_overfit_32(self) -> None:
        """Verify model can overfit 32 samples to >=95% accuracy."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = sanity_check_overfit_32(seed=2026, output_dir=tmp_dir)
            self.assertTrue(res["passed"], f"Overfit 32 failed with acc: {res['final_accuracy']}")
            self.assertGreaterEqual(res["final_accuracy"], 0.95)
            self.assertTrue((Path(tmp_dir) / "overfit_32.json").is_file())

    def test_centralized_training_artifacts(self) -> None:
        """Verify complete centralized training run generates all required audit artifacts."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            fixture_dir = Path(tmp_dir) / "fixture"
            create_synthetic_image_fixture(fixture_dir, num_real=20, num_fake=20, image_size=64, seed=42)
            samples = load_image_dataset(fixture_dir, seed=42)
            train_s, val_s, test_s = split_dataset(samples, seed=42)

            run_dir = Path(tmp_dir) / "run_output"
            results = train_centralized(
                train_samples=train_s,
                val_samples=val_s,
                test_samples=test_s,
                output_dir=run_dir,
                epochs=2,
                batch_size=8,
                seed=42,
            )

            # Assert all 5 mandatory artifact files exist
            self.assertTrue((run_dir / "checkpoint.pt").is_file())
            self.assertTrue((run_dir / "config.json").is_file())
            self.assertTrue((run_dir / "provenance.json").is_file())
            self.assertTrue((run_dir / "metrics.json").is_file())
            self.assertTrue((run_dir / "manifest.json").is_file())

            # Verify contents of metrics.json
            with open(run_dir / "metrics.json", "r", encoding="utf-8") as f:
                metrics = json.load(f)

            self.assertIn("best_epoch", metrics)
            self.assertIn("selected_threshold", metrics)
            self.assertIn("test_metrics", metrics)
            self.assertIn("accuracy", metrics["test_metrics"])
            self.assertIn("roc_auc", metrics["test_metrics"])
            self.assertIn("confusion_matrix", metrics["test_metrics"])


if __name__ == "__main__":
    unittest.main()
