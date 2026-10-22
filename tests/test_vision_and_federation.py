"""Tests for vision detection pipeline, federated engine, and live API endpoints."""

import io
import unittest
import numpy as np
from PIL import Image
import torch

from fedforge.api import create_fastapi_app
from fedforge.federated_engine import get_federated_engine
from fedforge.vision import DeepfakeDetector, MediaDetectionPipeline


class VisionPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = DeepfakeDetector()
        self.pipeline = MediaDetectionPipeline(model=self.detector)

    def test_model_forward_and_parameter_vector(self) -> None:
        x = torch.randn(2, 3, 224, 224)
        logits, mask = self.detector(x)
        self.assertEqual(logits.shape, (2, 1))
        self.assertEqual(mask.shape, (2, 1, 14, 14))

        params = self.detector.get_parameter_vector()
        self.assertGreater(len(params), 1000)
        self.detector.set_parameter_vector(params)

    def test_analyze_synthetic_image(self) -> None:
        # Create a synthetic 100x100 RGB image
        img = Image.new("RGB", (120, 120), color=(180, 120, 100))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        image_bytes = buf.getvalue()

        result = self.pipeline.analyze_image_bytes(image_bytes, filename="test.jpg")
        self.assertEqual(result["media_type"], "image")
        self.assertIn("verdict", result)
        self.assertIn("overall_score", result)
        self.assertIn("moderation_report", result)
        self.assertGreaterEqual(result["overall_score"], 0.0)
        self.assertLessEqual(result["overall_score"], 1.0)


class FederatedEngineTests(unittest.TestCase):
    def test_federated_step_and_security_demo(self) -> None:
        engine = get_federated_engine()
        engine.reset()
        self.assertEqual(engine.round, 0)
        self.assertEqual(len(engine.clients), 4)

        # Step 2 rounds
        state = engine.step_round()
        self.assertEqual(state["round"], 1)
        self.assertIn("accuracy", state["global_metrics"])
        self.assertIn("DeepFakes", state["generator_recall"])

        # Run 3-stage security experiment
        demo_results = engine.run_security_experiment()
        self.assertIn("clean", demo_results["stages"])
        self.assertIn("attack", demo_results["stages"])
        self.assertIn("defense", demo_results["stages"])
        self.assertEqual(demo_results["target_generator"], "Face2Face")


if __name__ == "__main__":
    unittest.main()
