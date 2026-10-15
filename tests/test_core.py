"""Focused offline tests for the FedForge CPU vertical slice."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from fedforge.attack import targeted_semantic_label_flip
from fedforge.data import (
    LeakageError,
    VideoRecord,
    assert_client_partition_no_leakage,
    assert_no_leakage,
    generate_synthetic_videos,
    partition_clients,
    split_train_test,
)
from fedforge.federation import ClientUpdate, fedavg, personalized_aggregate
from fedforge.metrics import auc, bootstrap_over_videos, ece, f1
from fedforge.payload import PayloadAuditError, audit_payload
from fedforge.robust import (
    CompatibilityError,
    coordinate_median,
    krum,
    norm_clip,
    trimmed_mean,
    validate_secagg_compatibility,
)
from fedforge.simulator import run_smoke_experiment


class DataInvariantTests(unittest.TestCase):
    def test_generation_is_deterministic(self) -> None:
        first = generate_synthetic_videos(seed=4, n_videos=12, source_count=4)
        second = generate_synthetic_videos(seed=4, n_videos=12, source_count=4)
        different = generate_synthetic_videos(seed=5, n_videos=12, source_count=4)
        self.assertEqual(first, second)
        self.assertNotEqual(first, different)

    def test_split_and_client_partition_are_source_disjoint(self) -> None:
        videos = generate_synthetic_videos(seed=8, n_videos=18, source_count=6)
        train, test = split_train_test(videos, test_fraction=0.3, seed=8)
        self.assertTrue(assert_no_leakage(train, test).ok)
        clients = partition_clients(train, 3, seed=8)
        assert_client_partition_no_leakage(clients)
        with self.assertRaises(LeakageError):
            assert_no_leakage((train[0],), (train[0],))


class AggregationTests(unittest.TestCase):
    def test_hand_cases(self) -> None:
        updates = [
            ClientUpdate("a", (1.0, 1.0), 1),
            ClientUpdate("b", (3.0, 3.0), 3),
        ]
        self.assertEqual(fedavg(updates), (2.5, 2.5))
        self.assertEqual(trimmed_mean([(0.0,), (1.0,), (2.0,), (100.0,)], trim_count=1), (1.5,))
        self.assertEqual(coordinate_median([(0.0, 10.0), (2.0, 2.0), (4.0, 0.0)]), (2.0, 2.0))
        selected = krum([(0.0,), (0.1,), (-0.1,), (0.2,), (10.0,)], byzantine_count=1)
        self.assertIn(selected, ((0.0,), (0.1,), (-0.1,), (0.2,)))
        clipped = norm_clip([(3.0, 4.0), (0.0, 0.0)], 1.0)
        self.assertAlmostEqual(clipped[0], 0.3)
        self.assertAlmostEqual(clipped[1], 0.4)

    def test_secagg_compatibility(self) -> None:
        self.assertTrue(validate_secagg_compatibility(True, "fedavg"))
        self.assertTrue(validate_secagg_compatibility(False, "krum"))
        with self.assertRaises(CompatibilityError):
            validate_secagg_compatibility(True, "trimmed_mean")
        with self.assertRaises(CompatibilityError):
            validate_secagg_compatibility(True, "norm_clip")
        self.assertTrue(validate_secagg_compatibility(True, "norm_clip", clipping_mode="client"))


class PersonalizationAndPayloadTests(unittest.TestCase):
    def test_local_heads_are_not_aggregated_or_emitted(self) -> None:
        updates = [
            ClientUpdate("a", (1.0, 2.0), 1, local_head=(10.0,)),
            ClientUpdate("b", (3.0, 4.0), 1, local_head=(-10.0,)),
        ]
        state = personalized_aggregate(updates)
        self.assertEqual(state.shared_parameters, (2.0, 3.0))
        self.assertEqual(state.local_heads, {"a": (10.0,), "b": (-10.0,)})
        safe_payload = updates[0].to_payload()
        self.assertTrue(audit_payload(safe_payload).valid)
        with self.assertRaises(PayloadAuditError):
            audit_payload(updates[0].to_payload(include_local_head=True))
        with self.assertRaises(PayloadAuditError):
            audit_payload({"filename": "frame-1.png"})
        with self.assertRaises(PayloadAuditError):
            audit_payload({"raw_media": b"not a media payload"})

    def test_semantic_flip_keeps_features_and_targets_generator(self) -> None:
        videos = generate_synthetic_videos(seed=3, n_videos=9, source_count=3)
        changed = targeted_semantic_label_flip(videos, target_generator="gan")
        for original, attacked in zip(videos, changed):
            self.assertEqual(
                tuple(frame.features for frame in original.frames),
                tuple(frame.features for frame in attacked.frames),
            )
            if original.generator == "gan":
                self.assertEqual(attacked.label, 0)
            else:
                self.assertEqual(original, attacked)


class MetricAndSmokeTests(unittest.TestCase):
    def test_metrics_and_video_bootstrap(self) -> None:
        self.assertEqual(auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertEqual(f1([1, 1, 0], [1, 0, 0]), 2 / 3)
        self.assertGreaterEqual(ece([0, 1], [0.1, 0.9]), 0.0)
        videos = generate_synthetic_videos(seed=2, n_videos=6, source_count=3)
        result = bootstrap_over_videos(videos, lambda sample: len(sample), n_resamples=10, seed=9)
        self.assertEqual(result.estimate, 6.0)
        self.assertEqual(len(result.samples), 10)

    def test_smoke_writes_reproducible_run_bundle(self) -> None:
        config = {
            "seed": 6,
            "n_videos": 18,
            "frames_per_video": 3,
            "n_features": 3,
            "source_count": 6,
            "n_clients": 3,
            "rounds": 2,
            "local_epochs": 1,
            "learning_rate": 0.2,
            "personalized_local_head": True,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            first_dir = root / "first"
            second_dir = root / "second"
            first = run_smoke_experiment(config_path, first_dir)
            second = run_smoke_experiment(config_path, second_dir)
            self.assertEqual(first.metrics, second.metrics)
            for run_dir in (first_dir, second_dir):
                self.assertTrue((run_dir / "provenance.json").is_file())
                self.assertTrue((run_dir / "metrics.jsonl").is_file())
                self.assertTrue((run_dir / "checkpoint.json").is_file())
                self.assertGreater((run_dir / "metrics.jsonl").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
