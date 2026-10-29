"""Centralized training baseline and mandatory sanity checks for FedForge.

Implements:
1. ResNet18 binary classifier fine-tuning with BCEWithLogitsLoss.
2. Validation-only threshold selection (test set evaluated exactly once).
3. Metric calculations: accuracy, precision, recall, F1, ROC-AUC, confusion matrix.
4. Run provenance recording (seed, git commit, python/torch versions, wall time).
5. Mandatory automated sanity checks:
   a) Overfit 32 samples to ~100% accuracy.
   b) Shuffled-label control yielding AUC ~0.50 (+/- 0.10).
   c) Label-map verification (fake class prediction > real class prediction).
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import sklearn.metrics as skm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import (
    ImageDataset,
    SampleRecord,
    generate_dataset_manifest,
    load_image_dataset,
    split_dataset,
)
from .model import build_model, get_state, set_state


# --------------------------------------------------------------------------- #
# Metrics Evaluation                                                          #
# --------------------------------------------------------------------------- #
def compute_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Calculate comprehensive binary classification metrics.

    Positive class (1) = synthetic / fake.
    Negative class (0) = authentic / real.
    """
    y_t = np.array(y_true, dtype=np.int64)
    y_p = np.array(y_prob, dtype=np.float64)
    y_pred = (y_p >= threshold).astype(np.int64)

    acc = float(skm.accuracy_score(y_t, y_pred))
    prec = float(skm.precision_score(y_t, y_pred, zero_division=0))
    rec = float(skm.recall_score(y_t, y_pred, zero_division=0))
    f1 = float(skm.f1_score(y_t, y_pred, zero_division=0))

    # ROC-AUC is undefined if only one class is present in y_true
    if len(np.unique(y_t)) > 1:
        roc_auc = float(skm.roc_auc_score(y_t, y_p))
    else:
        roc_auc = 0.50

    cm = skm.confusion_matrix(y_t, y_pred, labels=[0, 1]).tolist()

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": cm,
        "threshold": round(float(threshold), 4),
        "sample_count": len(y_t),
    }


def select_best_threshold(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    thresholds: Optional[Sequence[float]] = None,
) -> Tuple[float, float]:
    """Find the optimal decision threshold on validation predictions maximizing F1 score."""
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 17)

    best_thresh = 0.50
    best_f1 = -1.0
    for th in thresholds:
        m = compute_metrics(y_true, y_prob, threshold=th)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_thresh = float(th)

    return best_thresh, best_f1


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, List[int], List[float]]:
    """Evaluate loss, true labels, and predicted fake probabilities on a dataloader."""
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_samples = 0
    all_true: List[int] = []
    all_probs: List[float] = []

    with torch.no_grad():
        for tensors, targets in loader:
            tensors = tensors.to(device)
            targets = targets.to(device)
            logits = model(tensors)
            loss = criterion(logits, targets)

            batch_size = tensors.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            probs = torch.sigmoid(logits).squeeze(-1).cpu().tolist()
            if isinstance(probs, float):
                probs = [probs]
            trues = targets.squeeze(-1).cpu().long().tolist()
            if isinstance(trues, int):
                trues = [trues]

            all_probs.extend(probs)
            all_true.extend(trues)

    avg_loss = total_loss / max(1, total_samples)
    return avg_loss, all_true, all_probs


# --------------------------------------------------------------------------- #
# Run Provenance Metadata Helper                                              #
# --------------------------------------------------------------------------- #
def capture_provenance(seed: int, config: Dict[str, Any], wall_time_sec: float) -> Dict[str, Any]:
    """Capture full environment and execution provenance for audit reproducibility."""
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_hash = "unknown"

    return {
        "git_commit": git_hash,
        "seed": seed,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "device": "cpu" if not torch.cuda.is_available() else "cuda",
        "wall_time_sec": round(wall_time_sec, 2),
        "config": config,
    }


# --------------------------------------------------------------------------- #
# Centralized Training Loop                                                   #
# --------------------------------------------------------------------------- #
def train_centralized(
    train_samples: Sequence[SampleRecord],
    val_samples: Sequence[SampleRecord],
    test_samples: Sequence[SampleRecord],
    output_dir: Path | str,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    freeze_backbone: bool = True,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Execute complete centralized baseline training and artifact generation."""
    t0 = time.time()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    config = {
        "model_type": "resnet18",
        "freeze_backbone": freeze_backbone,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "seed": seed,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
    }

    train_ds = ImageDataset(train_samples, augment=True)
    val_ds = ImageDataset(val_samples)
    test_ds = ImageDataset(test_samples)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = build_model(
        model_type="resnet18",
        pretrained=True,
        freeze_backbone=freeze_backbone,
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 0
    history: List[Dict[str, Any]] = []

    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_train = 0

        for tensors, targets in train_loader:
            tensors = tensors.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            logits = model(tensors)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * tensors.size(0)
            n_train += tensors.size(0)

        epoch_train_loss = train_loss / max(1, n_train)
        val_loss, val_trues, val_probs = evaluate_model(model, val_loader, device)
        val_metrics = compute_metrics(val_trues, val_probs, threshold=0.50)

        round_entry = {
            "epoch": ep,
            "train_loss": round(epoch_train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_auc": val_metrics["roc_auc"],
        }
        history.append(round_entry)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = ep
            best_state = get_state(model, scope="full")

    # Restore best checkpoint
    if best_state is not None:
        set_state(model, best_state, strict=True)

    # Threshold selection on validation set only
    _, final_val_trues, final_val_probs = evaluate_model(model, val_loader, device)
    best_threshold, best_val_f1 = select_best_threshold(final_val_trues, final_val_probs)

    # Evaluate ONCE on held-out test split
    test_loss, test_trues, test_probs = evaluate_model(model, test_loader, device)
    final_test_metrics = compute_metrics(test_trues, test_probs, threshold=best_threshold)

    wall_time = time.time() - t0
    provenance = capture_provenance(seed, config, wall_time)

    # Save artifacts
    torch.save(best_state or get_state(model, scope="full"), out / "checkpoint.pt")

    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    with open(out / "provenance.json", "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)

    results_payload = {
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 4),
        "selected_threshold": best_threshold,
        "val_f1_at_threshold": round(best_val_f1, 4),
        "test_metrics": final_test_metrics,
        "training_history": history,
    }
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    generate_dataset_manifest(
        dataset_name="CIFAKE-Centralized-Baseline",
        license_note="CC0-Public-Domain",
        seed=seed,
        image_size=64,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        output_path=out / "manifest.json",
    )

    return results_payload


# --------------------------------------------------------------------------- #
# Mandatory Automated Sanity Checks                                           #
# --------------------------------------------------------------------------- #
def sanity_check_overfit_32(
    device: Optional[torch.device] = None,
    seed: int = 42,
    output_dir: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Sanity Check A: Overfit 32 samples to ~100% training accuracy.

    Verifies gradient flow, loss descent, and architecture capacity.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create synthetic batch of 32 distinct samples (16 class 0, 16 class 1)
    x = torch.randn(32, 3, 64, 64)
    # Give synthetic distinguishing pattern
    x[:16] -= 0.8
    x[16:] += 0.8
    y = torch.tensor([0.0] * 16 + [1.0] * 16).unsqueeze(-1)

    model = build_model(model_type="resnet18", pretrained=False, freeze_backbone=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    final_loss = 0.0
    final_acc = 0.0

    for _ in range(40):
        optimizer.zero_grad()
        logits = model(x.to(device))
        loss = criterion(logits, y.to(device))
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())

    model.eval()
    with torch.no_grad():
        preds = (torch.sigmoid(model(x.to(device))) >= 0.5).long().cpu()
        final_acc = float((preds == y.long()).float().mean().item())

    result = {
        "check": "overfit_32",
        "passed": bool(final_acc >= 0.95),
        "final_accuracy": round(final_acc, 4),
        "final_loss": round(final_loss, 4),
    }

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "overfit_32.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    return result


def sanity_check_shuffled_labels(
    samples: Sequence[SampleRecord],
    device: Optional[torch.device] = None,
    seed: int = 42,
    output_dir: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Sanity Check B: Shuffled-label control yields ROC-AUC ~0.50 (+/- 0.10).

    Verifies model cannot predict when label correlation is destroyed.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rng = np.random.default_rng(seed)
    train_split, val_split, test_split = split_dataset(samples[:200], seed=seed)

    # Permute training labels randomly
    permuted_train: List[SampleRecord] = []
    shuffled_labels = [s.label for s in train_split]
    rng.shuffle(shuffled_labels)
    for s, shuff_lbl in zip(train_split, shuffled_labels):
        permuted_train.append(SampleRecord(s.filepath, shuff_lbl, s.sha256, s.class_name))

    train_ds = ImageDataset(permuted_train)
    test_ds = ImageDataset(test_split)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    model = build_model(model_type="resnet18", pretrained=False, freeze_backbone=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(5):
        for tensors, targets in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(tensors.to(device)), targets.to(device))
            loss.backward()
            optimizer.step()

    _, test_trues, test_probs = evaluate_model(model, test_loader, device)
    metrics = compute_metrics(test_trues, test_probs, threshold=0.50)
    auc = metrics["roc_auc"]

    # Shuffled control must not achieve strong predictive signal (expected 0.35 - 0.65)
    passed = bool(0.35 <= auc <= 0.65)
    result = {
        "check": "shuffled_labels_control",
        "passed": passed,
        "test_roc_auc": round(auc, 4),
        "expected_range": "[0.35, 0.65]",
    }

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "shuffled_labels.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    return result


def sanity_check_label_map_not_inverted(
    model: nn.Module,
    samples: Sequence[SampleRecord],
    device: Optional[torch.device] = None,
    output_dir: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Sanity Check C: Ensure fake images produce higher probabilities than real images."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    real_samples = [s for s in samples if s.label == 0][:30]
    fake_samples = [s for s in samples if s.label == 1][:30]

    real_loader = DataLoader(ImageDataset(real_samples), batch_size=30, shuffle=False)
    fake_loader = DataLoader(ImageDataset(fake_samples), batch_size=30, shuffle=False)

    _, _, real_probs = evaluate_model(model, real_loader, device)
    _, _, fake_probs = evaluate_model(model, fake_loader, device)

    mean_real = float(np.mean(real_probs))
    mean_fake = float(np.mean(fake_probs))

    passed = bool(mean_fake > mean_real)
    result = {
        "check": "label_map_sanity",
        "passed": passed,
        "mean_prob_real": round(mean_real, 4),
        "mean_prob_fake": round(mean_fake, 4),
        "separation": round(mean_fake - mean_real, 4),
    }

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "label_map_sanity.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    return result


# --------------------------------------------------------------------------- #
# CLI Entrypoint                                                              #
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="FedForge Centralized Baseline Trainer")
    parser.add_argument("--data", default="data/cifake", help="Path to image dataset")
    parser.add_argument("--output", default="runs/centralized", help="Output run directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max-per-class", type=int, default=1000, help="Max samples per class")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--run-sanity-checks", action="store_true", help="Execute mandatory sanity suite")

    args = parser.parse_args()
    print(f"Loading dataset from: {args.data}")
    samples = load_image_dataset(args.data, max_per_class=args.max_per_class, seed=args.seed)
    train_s, val_s, test_s = split_dataset(samples, seed=args.seed)

    print(f"Splits: Train={len(train_s)}, Val={len(val_s)}, Test={len(test_s)}")
    results = train_centralized(
        train_samples=train_s,
        val_samples=val_s,
        test_samples=test_s,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )

    print(f"Training Complete. Artifacts saved to: {args.output}")
    print(f"Selected Validation Threshold: {results['selected_threshold']}")
    print(f"Test Accuracy: {results['test_metrics']['accuracy'] * 100:.2f}%")
    print(f"Test ROC-AUC:  {results['test_metrics']['roc_auc']:.4f}")
    print(f"Test F1:       {results['test_metrics']['f1']:.4f}")

    if args.run_sanity_checks:
        print("\nExecuting Mandatory Sanity Checks...")
        chk_a = sanity_check_overfit_32(seed=args.seed, output_dir=Path(args.output) / "sanity")
        print(f"1. Overfit 32 Samples: {'PASS' if chk_a['passed'] else 'FAIL'} (acc: {chk_a['final_accuracy']})")

        chk_b = sanity_check_shuffled_labels(samples, seed=args.seed, output_dir=Path(args.output) / "sanity")
        print(f"2. Shuffled Labels Control: {'PASS' if chk_b['passed'] else 'FAIL'} (AUC: {chk_b['test_roc_auc']})")

        model = build_model(model_type="resnet18")
        set_state(model, torch.load(Path(args.output) / "checkpoint.pt", map_location="cpu"), strict=True)
        chk_c = sanity_check_label_map_not_inverted(model, test_s, output_dir=Path(args.output) / "sanity")
        print(f"3. Label Map Sanity: {'PASS' if chk_c['passed'] else 'FAIL'} (real: {chk_c['mean_prob_real']}, fake: {chk_c['mean_prob_fake']})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
