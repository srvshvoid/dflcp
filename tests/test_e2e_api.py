"""End-to-end integration tests for FedForge live FastAPI application."""

import io
import unittest
from fastapi.testclient import TestClient
from PIL import Image

from fedforge.api.app import create_fastapi_app


class E2EApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = create_fastapi_app(runs_root="runs")
        cls.client = TestClient(cls.app)

    def test_health_endpoint(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_federation_state_and_step(self) -> None:
        res = self.client.get("/federation/state")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("round", data)
        self.assertIn("clients", data)

        step_res = self.client.post("/federation/step")
        self.assertEqual(step_res.status_code, 200)
        step_data = step_res.json()
        self.assertGreater(step_data["round"], 0)

    def test_detect_image_upload(self) -> None:
        # Generate in-memory dummy image
        img = Image.new("RGB", (100, 100), color=(200, 150, 120))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        response = self.client.post(
            "/detect",
            files={"file": ("test_face.jpg", buf, "image/jpeg")},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["media_type"], "image")
        self.assertIn("verdict", payload)
        self.assertIn("confidence", payload)
        self.assertIn("face_details", payload)
        self.assertIn("moderation_report", payload)

    def test_security_demo_endpoint(self) -> None:
        response = self.client.post("/federation/security-demo")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("stages", payload)
        self.assertIn("clean", payload["stages"])
        self.assertIn("attack", payload["stages"])
        self.assertIn("defense", payload["stages"])

    def test_client_model_update_endpoint(self) -> None:
        response = self.client.post(
            "/federation/client-update",
            json={
                "client_id": "client-0",
                "parameters": [0.0] * 513,
                "num_examples": 10,
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["accepted"])
        self.assertEqual(response.json()["update"]["bytes"], 2052)

    def test_client_model_update_rejects_wrong_parameter_count(self) -> None:
        response = self.client.post(
            "/federation/client-update",
            json={"client_id": "client-0", "parameters": [0.0], "num_examples": 10},
        )
        self.assertEqual(response.status_code, 400)

    def test_detect_returns_503_when_no_checkpoint_available(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as empty_dir:
            empty_app = create_fastapi_app(runs_root=empty_dir)
            empty_client = TestClient(empty_app)
            img = Image.new("RGB", (64, 64), color=(100, 100, 100))
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            buf.seek(0)
            res = empty_client.post(
                "/detect",
                files={"file": ("test.jpg", buf, "image/jpeg")},
            )
            self.assertEqual(res.status_code, 503)
            self.assertIn("No trained model checkpoint found", res.json()["detail"])

    def test_dashboard_root_serves_html(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("FedForge", response.text)


if __name__ == "__main__":
    unittest.main()
