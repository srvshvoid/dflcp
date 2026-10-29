"""Regression tests for live media model selection and calibration."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from fedforge.vision.detector import MediaDetectionPipeline


class DetectorRegressionTests(unittest.TestCase):
    def test_discovery_ignores_attack_runs_and_selects_highest_auc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, auc in (
                ("centralized", 0.81),
                ("defense_scaling_norm_clip", 0.86),
                ("attack_1mal_none", 0.99),
            ):
                run = root / name
                run.mkdir()
                torch.save({}, run / "checkpoint.pt")
                (run / "metrics.json").write_text(
                    json.dumps({"test_metrics": {"roc_auc": auc}}),
                    encoding="utf-8",
                )
                if "defense" in name:
                    (run / "config.json").write_text(
                        json.dumps({"attack_type": "update_scaling", "num_malicious": 1}),
                        encoding="utf-8",
                    )

            selected = MediaDetectionPipeline._discover_default_checkpoint(root)
            self.assertEqual(selected.parent.name, "centralized")

    def test_discovery_prefers_cross_dataset_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, auc in (("centralized", 0.90), ("cross_dataset", 0.80)):
                run = root / name
                run.mkdir()
                torch.save({}, run / "checkpoint.pt")
                (run / "metrics.json").write_text(
                    json.dumps({"test_metrics": {"roc_auc": auc}}),
                    encoding="utf-8",
                )
            (root / "cross_dataset" / "external_validation.json").write_text(
                json.dumps({"decided_accuracy": 0.84, "coverage": 0.78}),
                encoding="utf-8",
            )
            selected = MediaDetectionPipeline._discover_default_checkpoint(root)
            self.assertEqual(selected.parent.name, "cross_dataset")

    def test_explicit_upload_threshold_is_preserved(self) -> None:
        # Explicit thresholds are useful for controlled evaluation and must
        # not be silently changed by the deployment safety policy.
        with tempfile.TemporaryDirectory():
            class StubModel:
                def eval(self):
                    return self

            pipeline = MediaDetectionPipeline(model=StubModel(), threshold=0.25)
            self.assertEqual(pipeline.threshold, 0.25)

    def test_public_policy_abstains_on_borderline_score(self) -> None:
        class StubModel:
            def eval(self):
                return self

        pipeline = MediaDetectionPipeline(model=StubModel(), threshold=0.50)
        self.assertEqual(pipeline._classify_score(0.50)[0], "INCONCLUSIVE")
        self.assertEqual(pipeline._classify_score(0.80)[0], "LIKELY_SYNTHETIC")
        self.assertEqual(pipeline._classify_score(0.20)[0], "AUTHENTIC")

    def test_image_inference_uses_full_image_model_input(self) -> None:
        class CaptureModel:
            def __init__(self):
                self.shapes = []

            def eval(self):
                return self

            def predict_proba(self, tensor):
                self.shapes.append(tuple(tensor.shape))
                return 0.1, torch.zeros((1, 1, 8, 8))

        from PIL import Image
        import io

        model = CaptureModel()
        pipeline = MediaDetectionPipeline(model=model)
        buffer = io.BytesIO()
        Image.new("RGB", (320, 180), (100, 120, 140)).save(buffer, "JPEG")
        pipeline.analyze_image_bytes(buffer.getvalue())
        self.assertTrue(model.shapes)
        self.assertTrue(all(shape == (1, 3, 64, 64) for shape in model.shapes))


if __name__ == "__main__":
    unittest.main()
