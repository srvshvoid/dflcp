"""CLI script to run federated learning experiments across partitioning regimes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path if needed
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fedforge.ml.data import load_image_dataset, split_dataset
from fedforge.ml.fl import run_federated_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description="FedForge Real Federated Training CLI")
    parser.add_argument("--data", default="data/cifake", help="Dataset directory")
    parser.add_argument(
        "--extra-data",
        action="append",
        default=[],
        help="Additional labelled dataset root; may be supplied multiple times",
    )
    parser.add_argument("--output", default="runs/fl_iid", help="Output directory")
    parser.add_argument("--clients", type=int, default=5, help="Number of clients")
    parser.add_argument("--rounds", type=int, default=10, help="Number of FL rounds")
    parser.add_argument("--local-epochs", type=int, default=1, help="Local epochs per round")
    parser.add_argument("--batch-size", type=int, default=32, help="Local batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--partition", default="iid", choices=["iid", "dirichlet"], help="Partitioning scheme")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet concentration parameter")
    parser.add_argument("--mu", type=float, default=0.0, help="FedProx proximal parameter")
    parser.add_argument("--attack", default="none", choices=["none", "label_flip", "update_scaling"], help="Attack type")
    parser.add_argument("--malicious-fraction", type=float, default=0.0, help="Fraction of malicious clients (e.g. 0.2, 0.4)")
    parser.add_argument("--defense", default="none", choices=["none", "trimmed_mean", "coordinate_median", "krum", "norm_clip"], help="Defense algorithm")
    parser.add_argument("--max-per-class", type=int, default=1000, help="Max samples per class")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dp-clip-norm", type=float, default=None, help="Enable client update clipping for DP")
    parser.add_argument("--dp-noise-multiplier", type=float, default=0.0, help="Gaussian noise multiplier")

    args = parser.parse_args()

    print(f"Loading data from {args.data}...")
    samples = load_image_dataset(args.data, max_per_class=args.max_per_class, seed=args.seed)
    for index, extra_data in enumerate(args.extra_data, 1):
        extra_samples = load_image_dataset(
            extra_data,
            max_per_class=args.max_per_class,
            seed=args.seed + index,
        )
        samples.extend(extra_samples)
        print(f"Added {len(extra_samples)} samples from {extra_data}")
    train_s, val_s, test_s = split_dataset(samples, seed=args.seed)
    print(f"Dataset splits: Train={len(train_s)}, Val={len(val_s)}, Test={len(test_s)}")
    print(f"Configuring {args.clients} clients with {args.partition} partition (alpha={args.alpha})...")
    print(f"Security: attack={args.attack}, malicious_fraction={args.malicious_fraction}, defense={args.defense}")

    results = run_federated_experiment(
        train_samples=train_s,
        val_samples=val_s,
        test_samples=test_s,
        output_dir=args.output,
        num_clients=args.clients,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        partition_type=args.partition,
        alpha=args.alpha,
        mu=args.mu,
        malicious_fraction=args.malicious_fraction,
        attack_type=args.attack,
        defense=args.defense,
        seed=args.seed,
        dp_clip_norm=args.dp_clip_norm,
        dp_noise_multiplier=args.dp_noise_multiplier,
    )

    print(f"\nFederated Training Finished. Output saved to: {args.output}")
    print(f"Final Test Accuracy: {results['test_metrics']['accuracy'] * 100:.2f}%")
    print(f"Final Test ROC-AUC:  {results['test_metrics']['roc_auc']:.4f}")
    print(f"Final Test F1:       {results['test_metrics']['f1']:.4f}")
    print(f"Total Bytes Wire:    {results['total_bytes_communicated']:,} bytes")
    print(f"Total Wall Time:     {results['total_wall_time_sec']:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
