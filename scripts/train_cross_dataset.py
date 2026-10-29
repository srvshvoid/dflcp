"""Train a clean mixed-domain checkpoint for live image inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from fedforge.ml.data import deduplicate_samples, load_image_dataset, split_dataset
from fedforge.ml.train import train_centralized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cifake", type=Path, default=Path("data/cifake"))
    parser.add_argument(
        "--external",
        type=Path,
        default=Path("data/external/real_fake_images"),
    )
    parser.add_argument("--output", type=Path, default=Path("runs/cross_dataset"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--unfreeze-backbone",
        action="store_true",
        help="Fine-tune the pretrained ResNet backbone instead of training only its head.",
    )
    args = parser.parse_args()

    cifake = load_image_dataset(args.cifake, seed=42)
    external = load_image_dataset(args.external, seed=43)
    samples = deduplicate_samples(cifake + external)
    train, val, test = split_dataset(samples, seed=42)
    print(f"mixed samples: {len(samples)} train={len(train)} val={len(val)} test={len(test)}")
    result = train_centralized(
        train_samples=train,
        val_samples=val,
        test_samples=test,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        freeze_backbone=not args.unfreeze_backbone,
        seed=42,
    )
    print(result["test_metrics"])


if __name__ == "__main__":
    main()
