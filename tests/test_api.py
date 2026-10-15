"""Tests for the dependency-free API/replay boundary."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from fedforge.api import RunNotFoundError, RunRepository, score_smoke_features
from fedforge.simulator import run_smoke_experiment


class ApiBoundaryTests(unittest.TestCase):
    def test_repository_lists_and_reads_run_artifacts(self) -> None:
        config = Path("configs/smoke.json")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "runs" / "one"
            run_smoke_experiment(config, run_dir)
            repository = RunRepository(root / "runs")
            summaries = repository.list_runs()
            self.assertEqual(len(summaries), 1)
            run_id = summaries[0]["directory"]
            payload = repository.get_run(run_id)
            self.assertEqual(len(payload["metrics"]), 3)
            self.assertEqual(payload["evidence_status"], "synthetic-unverified")
            with self.assertRaises(RunNotFoundError):
                repository.get_run("../one")

    def test_numeric_smoke_score_reads_checkpoint(self) -> None:
        config = Path("configs/smoke.json")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            run_smoke_experiment(config, run_dir)
            result = score_smoke_features(
                [1.0, 0.0, 0.0, 0.0],
                checkpoint=run_dir / "checkpoint.json",
            )
            self.assertEqual(result["mode"], "numeric-smoke")
            self.assertIn(result["label"], {"real", "fake"})
            self.assertFalse(result["calibrated"])

    def test_manifest_is_explicitly_unmeasured(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = RunRepository(Path(temporary)).privacy_payload_manifest()
            self.assertFalse(manifest["measured_transport"])
            self.assertIn("raw_media", manifest["forbidden_fields"])


if __name__ == "__main__":
    unittest.main()
