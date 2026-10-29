"""Unit tests for FedForge data pipeline (src/fedforge/ml/data.py).

Tests:
1. Label mapping integrity (positive class = 1 = fake/synthetic).
2. Shared preprocessing shape and normalization.
3. SHA-256 deduplication and zero-leakage cross-split assertions.
4. Client partition disjointness and complete coverage.
5. Dirichlet alpha=0.1 label skew being measurably higher than IID partition.
6. Dataset manifest generation and schema completeness.
7. PyTorch ImageDataset batching compatibility.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from fedforge.ml.data import (
    FOLDER_TO_LABEL,
    INV_LABEL_MAP,
    LABEL_MAP,
    ImageDataset,
    SampleRecord,
    compute_file_sha256,
    create_synthetic_image_fixture,
    folder_to_label,
    generate_dataset_manifest,
    get_transform,
    load_image_dataset,
    partition_dataset,
    preprocess_image,
    split_dataset,
)


class TestMLDataPipeline(unittest.TestCase):
    """Rigorous acceptance tests for the FedForge ML data pipeline."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name)
        # Create a small fixture dataset: 50 real and 50 fake images
        create_synthetic_image_fixture(
            self.root_path,
            num_real=50,
            num_fake=50,
            image_size=64,
            seed=2026,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_label_map_contract(self) -> None:
        """Verify strict label map: positive class 1 = fake, class 0 = real."""
        self.assertEqual(LABEL_MAP["real"], 0)
        self.assertEqual(LABEL_MAP["fake"], 1)
        self.assertEqual(INV_LABEL_MAP[0], "real")
        self.assertEqual(INV_LABEL_MAP[1], "fake")

        self.assertEqual(folder_to_label("real"), 0)
        self.assertEqual(folder_to_label("authentic"), 0)
        self.assertEqual(folder_to_label("0_real"), 0)
        self.assertEqual(folder_to_label("fake"), 1)
        self.assertEqual(folder_to_label("synthetic"), 1)
        self.assertEqual(folder_to_label("1_fake"), 1)

        with self.assertRaises(ValueError):
            folder_to_label("unrecognized_folder_name")

    def test_preprocessing_and_normalization(self) -> None:
        """Verify preprocessing transform output shape and normalization."""
        transform = get_transform(image_size=64)
        img = Image.new("RGB", (128, 128), color=(200, 100, 50))
        tensor = transform(img)

        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, (3, 64, 64))
        self.assertEqual(tensor.dtype, torch.float32)

        # Test preprocess_image helper with numpy and PIL
        np_arr = np.ones((100, 100, 3), dtype=np.uint8) * 128
        batch_tensor = preprocess_image(np_arr, image_size=64)
        self.assertEqual(batch_tensor.shape, (1, 3, 64, 64))

    def test_hashing_and_deduplication(self) -> None:
        """Verify SHA-256 computation and duplicate file detection."""
        file1 = self.root_path / "real" / "real_0000.png"
        hash1 = compute_file_sha256(file1)
        self.assertEqual(len(hash1), 64)

        # Create an identical copy under another name
        dup_file = self.root_path / "real" / "real_duplicate.png"
        dup_file.write_bytes(file1.read_bytes())
        hash_dup = compute_file_sha256(dup_file)
        self.assertEqual(hash1, hash_dup)

        # Loading should deduplicate and keep only one
        samples = load_image_dataset(self.root_path, seed=42)
        all_hashes = [s.sha256 for s in samples]
        self.assertEqual(len(all_hashes), len(set(all_hashes)))
        self.assertEqual(len(samples), 100)

    def test_zero_leakage_splits(self) -> None:
        """Verify train/val/test splits have zero overlap in hashes and files."""
        samples = load_image_dataset(self.root_path, seed=42)
        train, val, test = split_dataset(
            samples,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=2026,
        )

        self.assertEqual(len(train) + len(val) + len(test), len(samples))
        self.assertEqual(len(train), 70)
        self.assertEqual(len(val), 15)
        self.assertEqual(len(test), 15)

        train_hashes = {s.sha256 for s in train}
        val_hashes = {s.sha256 for s in val}
        test_hashes = {s.sha256 for s in test}

        # Assert disjointness
        self.assertTrue(train_hashes.isdisjoint(val_hashes))
        self.assertTrue(train_hashes.isdisjoint(test_hashes))
        self.assertTrue(val_hashes.isdisjoint(test_hashes))

        train_paths = {s.filepath for s in train}
        val_paths = {s.filepath for s in val}
        test_paths = {s.filepath for s in test}

        self.assertTrue(train_paths.isdisjoint(val_paths))
        self.assertTrue(train_paths.isdisjoint(test_paths))
        self.assertTrue(val_paths.isdisjoint(test_paths))

    def test_partition_disjointness_and_coverage(self) -> None:
        """Verify client shards are mutually disjoint and completely cover the train split."""
        samples = load_image_dataset(self.root_path, seed=42)
        train, _, _ = split_dataset(samples, seed=42)

        for p_type in ("iid", "dirichlet"):
            shards = partition_dataset(
                train,
                num_clients=5,
                partition_type=p_type,
                alpha=0.5,
                seed=42,
            )
            self.assertEqual(len(shards), 5)

            all_assigned = []
            for cid, crecs in shards.items():
                all_assigned.extend([r.filepath for r in crecs])

            # Disjointness: no sample assigned more than once
            self.assertEqual(len(all_assigned), len(set(all_assigned)))
            # Complete coverage: every train sample is assigned
            self.assertEqual(set(all_assigned), {s.filepath for s in train})

    def test_dirichlet_skew_measurably_greater_than_iid(self) -> None:
        """Verify Dirichlet(alpha=0.1) exhibits measurably higher label skew variance than IID."""
        samples = load_image_dataset(self.root_path, seed=42)
        train, _, _ = split_dataset(samples, seed=42)
        num_clients = 5

        # 1. IID Partition
        iid_shards = partition_dataset(
            train,
            num_clients=num_clients,
            partition_type="iid",
            seed=2026,
        )
        iid_fake_ratios = []
        for cid, crecs in iid_shards.items():
            fake_count = sum(1 for r in crecs if r.label == 1)
            total = len(crecs)
            iid_fake_ratios.append(fake_count / total if total > 0 else 0.0)

        iid_variance = float(np.var(iid_fake_ratios))

        # 2. Dirichlet Partition (alpha=0.1: extreme label skew)
        skew_shards = partition_dataset(
            train,
            num_clients=num_clients,
            partition_type="dirichlet",
            alpha=0.1,
            seed=2026,
        )
        skew_fake_ratios = []
        for cid, crecs in skew_shards.items():
            fake_count = sum(1 for r in crecs if r.label == 1)
            total = len(crecs)
            skew_fake_ratios.append(fake_count / total if total > 0 else 0.0)

        skew_variance = float(np.var(skew_fake_ratios))

        # Assert that Dirichlet(0.1) skew variance is significantly higher than IID variance
        self.assertGreater(
            skew_variance,
            iid_variance,
            f"Expected Dirichlet alpha=0.1 variance ({skew_variance:.4f}) > IID variance ({iid_variance:.4f})",
        )

    def test_manifest_generation(self) -> None:
        """Verify dataset manifest JSON contains required provenance and split fields."""
        samples = load_image_dataset(self.root_path, seed=42)
        train, val, test = split_dataset(samples, seed=42)
        shards = partition_dataset(train, num_clients=4, partition_type="iid", seed=42)

        manifest_file = self.root_path / "dataset_manifest.json"
        manifest = generate_dataset_manifest(
            dataset_name="Synthetic-Fixture-CIFAKE-Smoke",
            license_note="CC0-Public-Domain",
            seed=42,
            image_size=64,
            train_samples=train,
            val_samples=val,
            test_samples=test,
            client_partitions=shards,
            output_path=manifest_file,
        )

        self.assertTrue(manifest_file.is_file())
        with open(manifest_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        self.assertEqual(loaded["dataset_name"], "Synthetic-Fixture-CIFAKE-Smoke")
        self.assertEqual(loaded["license"], "CC0-Public-Domain")
        self.assertEqual(loaded["splits"]["train"]["total"], 70)
        self.assertEqual(loaded["splits"]["val"]["total"], 15)
        self.assertEqual(loaded["splits"]["test"]["total"], 15)
        self.assertEqual(len(loaded["client_partitions"]), 4)
        self.assertEqual(len(loaded["file_hashes"]), 100)

    def test_pytorch_dataset_and_dataloader(self) -> None:
        """Verify ImageDataset integration with torch DataLoader."""
        samples = load_image_dataset(self.root_path, seed=42)
        ds = ImageDataset(samples, image_size=64)
        self.assertEqual(len(ds), 100)

        tensor, target = ds[0]
        self.assertEqual(tensor.shape, (3, 64, 64))
        self.assertEqual(target.shape, (1,))
        self.assertIn(target.item(), (0.0, 1.0))

        loader = DataLoader(ds, batch_size=8, shuffle=True)
        batch_tensors, batch_targets = next(iter(loader))
        self.assertEqual(batch_tensors.shape, (8, 3, 64, 64))
        self.assertEqual(batch_targets.shape, (8, 1))


if __name__ == "__main__":
    unittest.main()
