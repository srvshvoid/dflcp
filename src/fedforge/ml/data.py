"""Data pipeline for FedForge: label mapping, hashing, stratified splits, and Dirichlet partitions.

Single source of truth for:
1. Label mapping: positive class (1) = fake/synthetic, (0) = real/authentic.
2. Image preprocessing: RGB conversion, resize, ToTensor, Normalize with ImageNet stats.
3. Disjoint split verification: SHA-256 hash detection preventing cross-split leakage.
4. Non-IID client partitioning: IID vs Dirichlet label-skew with disjointness assertions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

# --------------------------------------------------------------------------- #
# Single Source of Truth for Class Labels                                     #
# --------------------------------------------------------------------------- #
LABEL_MAP: Dict[str, int] = {
    "real": 0,
    "fake": 1,
}
INV_LABEL_MAP: Dict[int, str] = {
    0: "real",
    1: "fake",
}

FOLDER_TO_LABEL: Dict[str, int] = {
    "real": 0,
    "authentic": 0,
    "0_real": 0,
    "cifar_real": 0,
    "fake": 1,
    "synthetic": 1,
    "1_fake": 1,
    "cifar_fake": 1,
}

# Strict assertions on label mapping contracts
assert LABEL_MAP["real"] == 0, "LABEL_MAP['real'] must equal 0 (authentic/organic class)"
assert LABEL_MAP["fake"] == 1, "LABEL_MAP['fake'] must equal 1 (synthetic/deepfake positive class)"


def folder_to_label(folder_name: str) -> int:
    """Resolve a directory name to an integer class label."""
    clean = folder_name.strip().lower()
    if clean in FOLDER_TO_LABEL:
        return FOLDER_TO_LABEL[clean]
    raise ValueError(
        f"Unknown folder name '{folder_name}'. Allowed folder aliases: {list(FOLDER_TO_LABEL.keys())}"
    )


# --------------------------------------------------------------------------- #
# Shared Preprocessing Function                                               #
# --------------------------------------------------------------------------- #
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


def get_transform(image_size: int = 64) -> T.Compose:
    """Return the canonical preprocessing transform shared across train, eval, and API inference."""
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])


def get_train_transform(image_size: int = 64) -> T.Compose:
    """Return a conservative augmentation transform for model training.

    CIFAKE images are small, so aggressive crops or geometric distortion can
    erase the artifacts the classifier needs to learn.  Horizontal flips and
    small photometric changes improve robustness while preserving those
    artifacts.  Validation, test, and API inference continue to use
    :func:`get_transform`.
    """
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.08, hue=0.02),
        T.ToTensor(),
        T.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])


def preprocess_image(
    image: Image.Image | np.ndarray,
    image_size: int = 64,
) -> torch.Tensor:
    """Preprocess a single PIL Image or NumPy array into a normalized (1, 3, H, W) tensor."""
    if isinstance(image, np.ndarray):
        # Convert BGR/Grayscale to RGB PIL Image
        if image.ndim == 2:
            pil_img = Image.fromarray(image).convert("RGB")
        elif image.ndim == 3 and image.shape[2] == 3:
            # Assume RGB unless otherwise specified
            pil_img = Image.fromarray(image).convert("RGB")
        else:
            raise ValueError(f"Unsupported image array shape: {image.shape}")
    elif isinstance(image, Image.Image):
        pil_img = image.convert("RGB")
    else:
        raise TypeError(f"Expected PIL.Image or np.ndarray, got {type(image)}")

    transform = get_transform(image_size=image_size)
    tensor = transform(pil_img)  # (3, H, W)
    return tensor.unsqueeze(0)   # (1, 3, H, W)


# --------------------------------------------------------------------------- #
# File Hashing and Data Integrity                                             #
# --------------------------------------------------------------------------- #
def compute_file_sha256(filepath: Path | str) -> str:
    """Compute deterministic SHA-256 hex digest for an image file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass(frozen=True)
class SampleRecord:
    """Immutable metadata record for a single dataset sample."""
    filepath: str
    label: int
    sha256: str
    class_name: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filepath": self.filepath,
            "label": self.label,
            "sha256": self.sha256,
            "class_name": self.class_name,
        }


# --------------------------------------------------------------------------- #
# Dataset Loader & Deduplication                                              #
# --------------------------------------------------------------------------- #
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_image_dataset(
    root_dir: Path | str,
    max_per_class: Optional[int] = None,
    seed: int = 42,
) -> List[SampleRecord]:
    """Scan directory structure, map folder labels, compute hashes, and deduplicate.

    Expected directory layout:
        root_dir/
            real/ (or authentic/) -> label 0
            fake/ (or synthetic/) -> label 1
    """
    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root directory '{root_dir}' does not exist.")

    samples_by_class: Dict[int, List[SampleRecord]] = {0: [], 1: []}
    seen_hashes: Set[str] = set()
    duplicate_count = 0

    file_label_pairs: List[Tuple[Path, int]] = []
    subdirs = [p for p in root.iterdir() if p.is_dir()]
    known_subdirs = [d for d in subdirs if d.name.lower() in FOLDER_TO_LABEL]

    if known_subdirs:
        for sdir in sorted(known_subdirs, key=lambda d: d.name):
            lbl = folder_to_label(sdir.name)
            for entry in os.scandir(sdir):
                if entry.is_file() and Path(entry.name).suffix.lower() in SUPPORTED_EXTENSIONS:
                    file_label_pairs.append((Path(entry.path), lbl))
    else:
        # Recursive walk for nested structures
        for dirpath, _, filenames in os.walk(root):
            curr_dir = Path(dirpath)
            folder_name = curr_dir.name
            try:
                lbl = folder_to_label(folder_name)
            except ValueError:
                try:
                    lbl = folder_to_label(curr_dir.parent.name)
                except ValueError:
                    continue
            for fname in sorted(filenames):
                if Path(fname).suffix.lower() in SUPPORTED_EXTENSIONS:
                    file_label_pairs.append((curr_dir / fname, lbl))

    for path, label in file_label_pairs:
        file_hash = compute_file_sha256(path)
        if file_hash in seen_hashes:
            duplicate_count += 1
            continue
        seen_hashes.add(file_hash)

        record = SampleRecord(
            filepath=str(path.resolve()),
            label=label,
            sha256=file_hash,
            class_name=INV_LABEL_MAP[label],
        )
        samples_by_class[label].append(record)

    # Deterministic stratified subsampling per class
    rng = np.random.default_rng(seed)
    final_samples: List[SampleRecord] = []
    for label in (0, 1):
        class_records = samples_by_class[label]
        # Sort by sha256 to ensure invariant initial ordering before shuffle
        class_records.sort(key=lambda r: r.sha256)
        if max_per_class is not None and len(class_records) > max_per_class:
            chosen_indices = rng.choice(len(class_records), size=max_per_class, replace=False)
            class_records = [class_records[i] for i in chosen_indices]
        final_samples.extend(class_records)

    # Final deterministic shuffle
    rng.shuffle(final_samples)
    return final_samples


def deduplicate_samples(samples: Sequence[SampleRecord]) -> List[SampleRecord]:
    """Remove duplicate content from a combined dataset while preserving labels.

    Loading separate sources deduplicates within each source only.  This helper
    is intended for merged datasets and rejects the same image appearing with
    conflicting labels instead of silently choosing one.
    """
    unique: Dict[str, SampleRecord] = {}
    for sample in samples:
        previous = unique.get(sample.sha256)
        if previous is not None:
            if previous.label != sample.label:
                raise ValueError(
                    f"Image hash {sample.sha256} appears with conflicting labels "
                    f"({previous.label} and {sample.label})"
                )
            continue
        unique[sample.sha256] = sample
    return list(unique.values())


# --------------------------------------------------------------------------- #
# Stratified Train/Val/Test Split with Zero-Leakage Assertion                #
# --------------------------------------------------------------------------- #
def split_dataset(
    samples: Sequence[SampleRecord],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[SampleRecord], List[SampleRecord], List[SampleRecord]]:
    """Create deterministic stratified train, validation, and test splits with zero leakage."""
    if not samples:
        raise ValueError("Cannot split an empty sample list.")
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-5:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    rng = np.random.default_rng(seed)
    target_train = int(round(train_ratio * len(samples)))
    target_val = int(round(val_ratio * len(samples)))
    target_test = len(samples) - target_train - target_val

    c0 = [s for s in samples if s.label == 0]
    c1 = [s for s in samples if s.label == 1]
    c0.sort(key=lambda s: s.sha256)
    c1.sort(key=lambda s: s.sha256)
    rng.shuffle(c0)
    rng.shuffle(c1)

    n0 = len(c0)
    n_train_0 = int(round(target_train * (n0 / len(samples))))
    n_val_0 = int(round(target_val * (n0 / len(samples))))
    n_test_0 = n0 - n_train_0 - n_val_0

    n_train_1 = target_train - n_train_0
    n_val_1 = target_val - n_val_0
    n_test_1 = target_test - n_test_0

    train_set = c0[:n_train_0] + c1[:n_train_1]
    val_set = c0[n_train_0: n_train_0 + n_val_0] + c1[n_train_1: n_train_1 + n_val_1]
    test_set = c0[n_train_0 + n_val_0:] + c1[n_train_1 + n_val_1:]

    # Strict Zero-Leakage Assertions by SHA-256 Hash and Filepath
    train_hashes = {s.sha256 for s in train_set}
    val_hashes = {s.sha256 for s in val_set}
    test_hashes = {s.sha256 for s in test_set}

    assert train_hashes.isdisjoint(val_hashes), "Zero-Leakage Violation: Train and Val share duplicate hashes!"
    assert train_hashes.isdisjoint(test_hashes), "Zero-Leakage Violation: Train and Test share duplicate hashes!"
    assert val_hashes.isdisjoint(test_hashes), "Zero-Leakage Violation: Val and Test share duplicate hashes!"

    train_paths = {s.filepath for s in train_set}
    val_paths = {s.filepath for s in val_set}
    test_paths = {s.filepath for s in test_set}

    assert train_paths.isdisjoint(val_paths), "Zero-Leakage Violation: Train and Val share duplicate files!"
    assert train_paths.isdisjoint(test_paths), "Zero-Leakage Violation: Train and Test share duplicate files!"
    assert val_paths.isdisjoint(test_paths), "Zero-Leakage Violation: Val and Test share duplicate files!"

    assert len(train_set) + len(val_set) + len(test_set) == len(samples), "Split counts do not sum to total samples!"

    # Deterministic final shuffle of each split
    rng.shuffle(train_set)
    rng.shuffle(val_set)
    rng.shuffle(test_set)

    return train_set, val_set, test_set


# --------------------------------------------------------------------------- #
# Client Partitioner: IID and Dirichlet Label-Skew                           #
# --------------------------------------------------------------------------- #
def partition_dataset(
    train_samples: Sequence[SampleRecord],
    num_clients: int,
    partition_type: str = "iid",
    alpha: float = 0.5,
    seed: int = 42,
) -> Dict[str, List[SampleRecord]]:
    """Partition training samples across simulated clients with strict disjointness.

    Args:
        train_samples: The train split to shard across clients.
        num_clients: Total number of client platforms.
        partition_type: 'iid' for uniform random, 'dirichlet' for non-IID label skew.
        alpha: Dirichlet concentration parameter (lower = more skewed).
        seed: Random seed for reproducibility.

    Returns:
        dict mapping client_id -> list of SampleRecord.
    """
    if num_clients < 1:
        raise ValueError("num_clients must be at least 1")
    if not train_samples:
        raise ValueError("train_samples cannot be empty")

    rng = np.random.default_rng(seed)
    client_ids = [f"client_{i}" for i in range(num_clients)]
    shards: Dict[str, List[SampleRecord]] = {cid: [] for cid in client_ids}

    if partition_type == "iid":
        # Uniform random partition
        shuffled = list(train_samples)
        shuffled.sort(key=lambda s: s.sha256)
        rng.shuffle(shuffled)
        chunks = np.array_split(shuffled, num_clients)
        for cid, chunk in zip(client_ids, chunks):
            shards[cid] = list(chunk)

    elif partition_type == "dirichlet":
        # Dirichlet label skew partitioning
        if alpha <= 0:
            raise ValueError(f"Dirichlet alpha must be positive, got {alpha}")

        # Group sample indices by class
        for label in (0, 1):
            class_records = [s for s in train_samples if s.label == label]
            class_records.sort(key=lambda s: s.sha256)
            rng.shuffle(class_records)

            n_class = len(class_records)
            if n_class == 0:
                continue

            # Sample Dirichlet proportions
            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            counts = (proportions * n_class).astype(int)

            # Fix rounding difference so sum(counts) == n_class
            diff = n_class - counts.sum()
            for i in range(abs(diff)):
                if diff > 0:
                    counts[i % num_clients] += 1
                elif diff < 0:
                    idx = np.argmax(counts)
                    counts[idx] -= 1

            start_idx = 0
            for cid, cnt in zip(client_ids, counts):
                shards[cid].extend(class_records[start_idx: start_idx + cnt])
                start_idx += cnt
    else:
        raise ValueError(f"Unknown partition_type '{partition_type}'. Choose 'iid' or 'dirichlet'.")

    # Strict Partition Disjointness and Full Coverage Assertions
    all_partitioned_paths: List[str] = []
    for cid, client_records in shards.items():
        all_partitioned_paths.extend([r.filepath for r in client_records])

    assert len(all_partitioned_paths) == len(train_samples), (
        f"Partitioned count ({len(all_partitioned_paths)}) does not match train sample count ({len(train_samples)})!"
    )
    assert len(set(all_partitioned_paths)) == len(train_samples), (
        "Client partition shards are NOT mutually disjoint!"
    )
    assert set(all_partitioned_paths) == {s.filepath for s in train_samples}, (
        "Client partition shards do not completely cover the training split!"
    )

    return shards


# --------------------------------------------------------------------------- #
# Dataset Manifest Generator                                                  #
# --------------------------------------------------------------------------- #
def generate_dataset_manifest(
    dataset_name: str,
    license_note: str,
    seed: int,
    image_size: int,
    train_samples: Sequence[SampleRecord],
    val_samples: Sequence[SampleRecord],
    test_samples: Sequence[SampleRecord],
    client_partitions: Optional[Dict[str, Sequence[SampleRecord]]] = None,
    output_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Generate and optionally save an immutable JSON dataset manifest."""
    def count_classes(recs: Sequence[SampleRecord]) -> Dict[str, int]:
        c0 = sum(1 for r in recs if r.label == 0)
        c1 = sum(1 for r in recs if r.label == 1)
        return {"total": len(recs), "real": c0, "fake": c1}

    manifest: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "license": license_note,
        "seed": seed,
        "image_size": image_size,
        "splits": {
            "train": count_classes(train_samples),
            "val": count_classes(val_samples),
            "test": count_classes(test_samples),
        },
        "total_samples": len(train_samples) + len(val_samples) + len(test_samples),
    }

    if client_partitions:
        manifest["client_partitions"] = {
            cid: count_classes(crecs) for cid, crecs in client_partitions.items()
        }

    manifest["file_hashes"] = {
        r.filepath: {"sha256": r.sha256, "label": r.label, "class": r.class_name}
        for r in list(train_samples) + list(val_samples) + list(test_samples)
    }

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)

    return manifest


# --------------------------------------------------------------------------- #
# PyTorch Dataset Interface                                                   #
# --------------------------------------------------------------------------- #
class ImageDataset(Dataset):
    """PyTorch Dataset yielding (normalized_tensor, binary_target)."""

    def __init__(
        self,
        samples: Sequence[SampleRecord],
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        image_size: int = 64,
        augment: bool = False,
    ) -> None:
        self.samples = list(samples)
        self.transform = transform or (
            get_train_transform(image_size=image_size)
            if augment
            else get_transform(image_size=image_size)
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        record = self.samples[idx]
        img = Image.open(record.filepath).convert("RGB")
        tensor = self.transform(img)
        target = torch.tensor([float(record.label)], dtype=torch.float32)
        return tensor, target


# --------------------------------------------------------------------------- #
# Deterministic Synthetic Fixture Generator (for fast unit testing & CI)      #
# --------------------------------------------------------------------------- #
def create_synthetic_image_fixture(
    root_dir: Path | str,
    num_real: int = 40,
    num_fake: int = 40,
    image_size: int = 64,
    seed: int = 42,
) -> Path:
    """Generate deterministic synthetic images for test fixtures and smoke runs."""
    root = Path(root_dir)
    real_dir = root / "real"
    fake_dir = root / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    # Real images: smooth spatial gradients + subtle low-amplitude noise
    for i in range(num_real):
        x = np.linspace(0, 1, image_size)
        y = np.linspace(0, 1, image_size)
        xx, yy = np.meshgrid(x, y)
        phase = rng.uniform(0, np.pi)
        r = 0.5 + 0.4 * np.sin(2 * np.pi * xx + phase)
        g = 0.5 + 0.4 * np.cos(2 * np.pi * yy + phase)
        b = 0.5 + 0.4 * np.sin(2 * np.pi * (xx + yy))
        arr = np.stack([r, g, b], axis=-1)
        arr = np.clip(arr * 255.0 + rng.normal(0, 3, arr.shape), 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        img.save(real_dir / f"real_{i:04d}.png")

    # Fake images: high-frequency grid artifacts / checkerboard patterns
    for i in range(num_fake):
        grid_freq = rng.choice([4, 8, 16])
        x = (np.arange(image_size) // grid_freq) % 2
        y = (np.arange(image_size) // grid_freq) % 2
        xx, yy = np.meshgrid(x, y)
        pattern = (xx ^ yy).astype(np.float32)
        r = pattern * 0.8 + 0.1
        g = (1.0 - pattern) * 0.7 + 0.15
        b = np.full_like(pattern, 0.5)
        arr = np.stack([r, g, b], axis=-1)
        arr = np.clip(arr * 255.0 + rng.normal(0, 2, arr.shape), 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        img.save(fake_dir / f"fake_{i:04d}.png")

    return root
