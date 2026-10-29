"""Unit tests for FedForge adversarial attacks and Byzantine robust defenses.

Tests:
1. Numerical agreement between robust_torch and pure-Python robust.py primitives.
2. Byzantine outlier rejection in a constructed adversary setting.
3. Label-flipping attack (symmetrical and targeted).
4. Parameter update-scaling attack.
"""

from __future__ import annotations

import unittest
import numpy as np
import torch

from fedforge.federation import ClientUpdate, fedavg
from fedforge.ml.attack import apply_label_flip, apply_update_scaling
from fedforge.ml.data import SampleRecord
from fedforge.ml.robust_torch import (
    get_robust_aggregator,
    torch_coordinate_median,
    torch_krum,
    torch_norm_clip,
    torch_trimmed_mean,
)
from fedforge.robust import coordinate_median, krum, norm_clip, trimmed_mean


class TestMLAttacksAndDefenses(unittest.TestCase):
    """Test suite for adversarial threat models and Byzantine robust aggregators."""

    def test_numerical_agreement_with_robust_py(self) -> None:
        """Verify torch-based robust aggregators match pure-Python robust.py exactly."""
        rng = np.random.default_rng(2026)
        num_clients = 7
        dim = 50

        # Generate random weight updates
        raw_updates = [tuple(float(v) for v in rng.normal(0, 1, dim)) for _ in range(num_clients)]
        tensor_updates = torch.tensor(raw_updates, dtype=torch.float32)

        # 1. Coordinate Median Agreement
        py_med = coordinate_median(raw_updates)
        th_med = torch_coordinate_median(tensor_updates).tolist()
        for p, t in zip(py_med, th_med):
            self.assertAlmostEqual(p, t, places=5)

        # 2. Trimmed Mean Agreement
        py_trim = trimmed_mean(raw_updates, trim_fraction=0.2)
        th_trim = torch_trimmed_mean(tensor_updates, trim_fraction=0.2).tolist()
        for p, t in zip(py_trim, th_trim):
            self.assertAlmostEqual(p, t, places=5)

        # 3. Krum Agreement (K=7, f=1: K > 2*1 + 2 = 4)
        py_krum_res = krum(raw_updates, byzantine_count=1)
        th_krum_res = torch_krum(tensor_updates, byzantine_count=1).tolist()
        for p, t in zip(py_krum_res, th_krum_res):
            self.assertAlmostEqual(p, t, places=5)

    def test_outlier_rejection_in_constructed_adversary(self) -> None:
        """Verify robust defenses reject a massive poisoning outlier while FedAvg collapses."""
        # 4 honest clients near [1.0, 1.0, 1.0]
        honest = [
            ClientUpdate("c0", (1.01, 0.99, 1.00), 100),
            ClientUpdate("c1", (0.98, 1.02, 1.01), 100),
            ClientUpdate("c2", (1.00, 1.01, 0.99), 100),
            ClientUpdate("c3", (1.02, 0.98, 1.00), 100),
        ]
        # 1 Byzantine attacker pushing massive outlier [100.0, 200.0, 300.0]
        malicious = ClientUpdate("c4_byz", (100.0, 200.0, 300.0), 100)
        cohort = honest + [malicious]

        # 1. Standard FedAvg is corrupted by the outlier
        corrupted_avg = fedavg(cohort)
        self.assertGreater(corrupted_avg[0], 20.0)
        self.assertGreater(corrupted_avg[1], 40.0)

        # 2. Coordinate Median rejects the outlier
        median_agg = get_robust_aggregator("coordinate_median")
        med_res = median_agg(cohort)
        for val in med_res:
            self.assertAlmostEqual(val, 1.0, delta=0.05)

        # 3. Trimmed Mean rejects the outlier
        trim_agg = get_robust_aggregator("trimmed_mean")
        trim_res = trim_agg(cohort)
        for val in trim_res:
            self.assertAlmostEqual(val, 1.0, delta=0.05)

        # 4. Krum selects an honest candidate
        krum_agg = get_robust_aggregator("krum", byzantine_count=1)
        krum_res = krum_agg(cohort)
        for val in krum_res:
            self.assertAlmostEqual(val, 1.0, delta=0.05)

        # 5. Norm Clip bounds the outlier influence
        clip_agg = get_robust_aggregator("norm_clip", max_norm=2.0)
        clip_res = clip_agg(cohort)
        # Without clipping, average was > 20. With clipping, norm is strongly bounded (< 2.0)
        self.assertLess(clip_res[0], 2.0)
        self.assertLess(clip_res[1], 2.0)

    def test_label_flip_attack(self) -> None:
        """Verify symmetrical and targeted label flipping."""
        samples = [
            SampleRecord("f0.jpg", 0, "hash0", "real"),
            SampleRecord("f1.jpg", 1, "hash1", "fake"),
            SampleRecord("f2.jpg", 0, "hash2", "real"),
            SampleRecord("f3.jpg", 1, "hash3", "fake"),
        ]

        # 1. Symmetrical flip (all 0->1, 1->0)
        flipped = apply_label_flip(samples, flip_rate=1.0, targeted_class=None)
        self.assertEqual([s.label for s in flipped], [1, 0, 1, 0])

        # 2. Targeted flip: only class 1 (fake) flipped to 0 (real)
        targeted = apply_label_flip(samples, flip_rate=1.0, targeted_class=1)
        self.assertEqual([s.label for s in targeted], [0, 0, 0, 0])

    def test_update_scaling_attack(self) -> None:
        """Verify update scaling multiplies perturbation delta by scale factor."""
        global_w = (1.0, 2.0, 3.0)
        local_w = (1.1, 2.2, 3.3)  # delta = [0.1, 0.2, 0.3]
        scale = 5.0

        poisoned = apply_update_scaling(local_w, global_w, scale_factor=scale)
        # Expected: 1.0 + 5*0.1 = 1.5, 2.0 + 5*0.2 = 3.0, 3.0 + 5*0.3 = 4.5
        expected = (1.5, 3.0, 4.5)
        for p, exp in zip(poisoned, expected):
            self.assertAlmostEqual(p, exp, places=5)


if __name__ == "__main__":
    unittest.main()
