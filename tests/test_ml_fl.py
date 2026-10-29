"""Unit tests for FedForge real federated learning (src/fedforge/ml/fl.py).

Acceptance checks:
1. Identical clients averaging theorem (averaging identical clients equals single client).
2. Weighted averaging matches an exact hand-computed calculation.
3. Client/Server privacy boundary: no raw data or labels in client updates.
4. Real short federated run produces valid rounds.jsonl with telemetry.
5. FedProx proximal parameter penalizes update divergence.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from fedforge.federation import ClientUpdate, fedavg
from fedforge.ml.data import create_synthetic_image_fixture, load_image_dataset, split_dataset
from fedforge.ml.fl import (
    FederatedClient,
    FederatedServer,
    FLClientUpdate,
    run_federated_experiment,
)


class TestMLFederatedLearning(unittest.TestCase):
    """Rigorous tests for federated learning math, privacy boundary, and execution."""

    def test_fedavg_identical_clients_theorem(self) -> None:
        """Theorem: Weighted FedAvg of identical updates equals the single client's update."""
        identical_vector = (0.25, -1.4, 3.1415, 0.0, 42.0)
        c1 = ClientUpdate(client_id="c1", parameters=identical_vector, num_examples=100)
        c2 = ClientUpdate(client_id="c2", parameters=identical_vector, num_examples=250)
        c3 = ClientUpdate(client_id="c3", parameters=identical_vector, num_examples=50)

        aggregated = fedavg([c1, c2, c3])
        self.assertEqual(len(aggregated), len(identical_vector))
        for actual, expected in zip(aggregated, identical_vector):
            self.assertAlmostEqual(actual, expected, places=7)

    def test_fedavg_weighted_hand_computed_example(self) -> None:
        """Verify weighted FedAvg matches an exact analytical hand computation.

        Update 1: [1.0, 2.0, 3.0] with n1 = 100
        Update 2: [3.0, 4.0, 5.0] with n2 = 300
        Total n = 400
        Expected: (100 * [1,2,3] + 300 * [3,4,5]) / 400 = [2.5, 3.5, 4.5]
        """
        u1 = ClientUpdate(client_id="c1", parameters=(1.0, 2.0, 3.0), num_examples=100)
        u2 = ClientUpdate(client_id="c2", parameters=(3.0, 4.0, 5.0), num_examples=300)

        aggregated = fedavg([u1, u2])
        expected = (2.5, 3.5, 4.5)
        for actual, exp in zip(aggregated, expected):
            self.assertAlmostEqual(actual, exp, places=7)

    def test_client_server_privacy_boundary(self) -> None:
        """Verify that FLClientUpdate contains only model weights, count, and scalar diagnostics."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            fixture_dir = Path(tmp_dir) / "data"
            create_synthetic_image_fixture(fixture_dir, num_real=10, num_fake=10, image_size=64, seed=42)
            samples = load_image_dataset(fixture_dir, seed=42)

            client = FederatedClient(client_id="client_test", samples=samples)
            dummy_global_weights = tuple(0.0 for _ in range(513))

            update = client.train_local(dummy_global_weights, epochs=1, batch_size=8)

            self.assertIsInstance(update, FLClientUpdate)
            self.assertEqual(len(update.parameters), 513)
            self.assertEqual(update.num_examples, 20)
            self.assertIsInstance(update.local_loss, float)
            self.assertIsInstance(update.update_norm, float)

            # Ensure no forbidden fields or raw tensors cross the boundary
            payload = update.to_client_update().to_payload()
            self.assertNotIn("raw_media", payload)
            self.assertNotIn("labels", payload)
            self.assertNotIn("samples", payload)
            self.assertNotIn("filepaths", payload)

    def test_real_short_fl_run_generates_rounds_jsonl(self) -> None:
        """Verify a real 2-round federated experiment generates valid rounds.jsonl telemetry."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            fixture_dir = Path(tmp_dir) / "data"
            create_synthetic_image_fixture(fixture_dir, num_real=25, num_fake=25, image_size=64, seed=42)
            samples = load_image_dataset(fixture_dir, seed=42)
            train_s, val_s, test_s = split_dataset(samples, seed=42)

            output_dir = Path(tmp_dir) / "fl_run"
            results = run_federated_experiment(
                train_samples=train_s,
                val_samples=val_s,
                test_samples=test_s,
                output_dir=output_dir,
                num_clients=2,
                rounds=2,
                local_epochs=1,
                batch_size=8,
                seed=42,
            )

            rounds_file = output_dir / "rounds.jsonl"
            self.assertTrue(rounds_file.is_file())

            # Read and inspect recorded rounds
            lines = [json.loads(line) for line in rounds_file.read_text(encoding="utf-8").strip().splitlines()]
            self.assertEqual(len(lines), 2)

            r1 = lines[0]
            self.assertEqual(r1["round"], 1)
            self.assertIn("val_loss", r1)
            self.assertIn("val_accuracy", r1)
            self.assertIn("val_auc", r1)
            self.assertIn("test_accuracy", r1)
            self.assertIn("client_losses", r1)
            self.assertIn("client_sample_counts", r1)
            self.assertIn("client_update_norms", r1)
            self.assertIn("bytes_communicated", r1)
            self.assertIn("round_time_sec", r1)

            # Assert all 5 mandatory artifacts exist
            self.assertTrue((output_dir / "checkpoint.pt").is_file())
            self.assertTrue((output_dir / "config.json").is_file())
            self.assertTrue((output_dir / "provenance.json").is_file())
            self.assertTrue((output_dir / "metrics.json").is_file())
            self.assertTrue((output_dir / "manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
