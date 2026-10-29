"""Real federated learning execution engine for FedForge (FedAvg and FedProx).

Features:
1. Strict Client/Server separation: zero raw data or labels cross the boundary.
2. Local training with optional FedProx proximal penalty (mu).
3. Weighted FedAvg aggregation over parameter state vectors.
4. Comprehensive per-round telemetry: global val/test metrics, client losses,
   L2 update norms, measured communication bytes, and wall time recorded to rounds.jsonl.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..federation import ClientUpdate, fedavg as core_fedavg
from .data import (
    ImageDataset,
    SampleRecord,
    generate_dataset_manifest,
    load_image_dataset,
    partition_dataset,
    split_dataset,
)
from .model import (
    FEDERATION_CONTRACT,
    build_model,
    get_state,
    get_state_schema,
    set_state,
    state_to_vector,
    vector_to_state,
)
from .train import capture_provenance, compute_metrics, evaluate_model, select_best_threshold
from ..security import clip_and_noise


# --------------------------------------------------------------------------- #
# Client Update Contract (Strict Privacy Boundary)                           #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FLClientUpdate:
    """Serializable client update payload returned across the privacy boundary.

    Contains exclusively the updated weight vector, sample count, and training diagnostics.
    Raw image tensors, file paths, and individual labels are strictly excluded.
    """
    client_id: str
    parameters: Tuple[float, ...]
    num_examples: int
    local_loss: float
    update_norm: float

    def to_client_update(self) -> ClientUpdate:
        """Convert to repository's core ClientUpdate dataclass."""
        return ClientUpdate(
            client_id=self.client_id,
            parameters=self.parameters,
            num_examples=self.num_examples,
        )


# --------------------------------------------------------------------------- #
# Federated Client                                                            #
# --------------------------------------------------------------------------- #
class FederatedClient:
    """Client platform holding local private data shards."""

    def __init__(
        self,
        client_id: str,
        samples: Sequence[SampleRecord],
        device: Optional[torch.device] = None,
        freeze_backbone: bool = True,
    ) -> None:
        self.client_id = client_id
        self.samples = list(samples)
        self.device = device or torch.device("cpu")
        self.freeze_backbone = freeze_backbone
        self.dataset = ImageDataset(self.samples)

        # Local model instance with shared feature backbone
        self.model = build_model(
            model_type="resnet18",
            pretrained=True,
            freeze_backbone=freeze_backbone,
        ).to(self.device)

        head_state = get_state(self.model, scope="head")
        self.schema = get_state_schema(head_state)

    def train_local(
        self,
        global_weights: Sequence[float],
        epochs: int = 1,
        lr: float = 1e-3,
        batch_size: int = 32,
        mu: float = 0.0,
        seed: int = 42,
        dp_clip_norm: float | None = None,
        dp_noise_multiplier: float = 0.0,
    ) -> FLClientUpdate:
        """Train locally on private shard for E epochs starting from global weights."""
        torch.manual_seed(seed)
        np.random.seed(seed)

        # 1. Load incoming global weights into local model head
        new_head_state = vector_to_state(global_weights, self.schema)
        set_state(self.model, new_head_state, strict=False)

        # Tensor representation of global weights for proximal term
        global_vector_tensor = torch.tensor(global_weights, dtype=torch.float32, device=self.device)

        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for _ in range(epochs):
            for tensors, targets in loader:
                tensors = tensors.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()

                logits = self.model(tensors)
                loss = criterion(logits, targets)

                # FedProx proximal regularization: (mu / 2) * || w - w_global ||^2
                if mu > 0.0:
                    current_head_state = get_state(self.model, scope="head")
                    current_vector = torch.tensor(
                        state_to_vector(current_head_state),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    prox_loss = (mu / 2.0) * torch.norm(current_vector - global_vector_tensor) ** 2
                    loss = loss + prox_loss

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * tensors.size(0)
                total_samples += tensors.size(0)

        # 2. Extract local updated weights
        local_head_state = get_state(self.model, scope="head")
        local_weights = state_to_vector(local_head_state)
        if dp_clip_norm is not None:
            local_weights = clip_and_noise(
                local_weights,
                global_weights,
                clip_norm=dp_clip_norm,
                noise_multiplier=dp_noise_multiplier,
            )

        # 3. Compute update L2 norm || w_local - w_global ||_2
        update_norm = float(
            math.sqrt(sum((l - g) ** 2 for l, g in zip(local_weights, global_weights)))
        )
        avg_loss = total_loss / max(1, total_samples)

        return FLClientUpdate(
            client_id=self.client_id,
            parameters=local_weights,
            num_examples=len(self.samples),
            local_loss=round(avg_loss, 4),
            update_norm=round(update_norm, 4),
        )


# --------------------------------------------------------------------------- #
# Federated Server                                                            #
# --------------------------------------------------------------------------- #
class FederatedServer:
    """Orchestrator managing communication rounds, aggregation, and global evaluation."""

    def __init__(
        self,
        val_samples: Sequence[SampleRecord],
        test_samples: Sequence[SampleRecord],
        device: Optional[torch.device] = None,
        freeze_backbone: bool = True,
    ) -> None:
        self.val_samples = list(val_samples)
        self.test_samples = list(test_samples)
        self.device = device or torch.device("cpu")
        self.freeze_backbone = freeze_backbone

        self.global_model = build_model(
            model_type="resnet18",
            pretrained=True,
            freeze_backbone=freeze_backbone,
        ).to(self.device)

        self.val_loader = DataLoader(ImageDataset(self.val_samples), batch_size=32, shuffle=False)
        self.test_loader = DataLoader(ImageDataset(self.test_samples), batch_size=32, shuffle=False)

        initial_head = get_state(self.global_model, scope="head")
        self.schema = get_state_schema(initial_head)
        self.global_weights = state_to_vector(initial_head)

    def aggregate_updates(
        self,
        client_updates: Sequence[FLClientUpdate],
        custom_aggregator: Optional[Callable[[Sequence[ClientUpdate]], Tuple[float, ...]]] = None,
    ) -> Tuple[float, ...]:
        """Aggregate client parameter updates into new global weights."""
        core_updates = [u.to_client_update() for u in client_updates]
        if custom_aggregator is not None:
            new_weights = custom_aggregator(core_updates)
        else:
            # Canonical sample-weighted FedAvg
            new_weights = core_fedavg(core_updates)

        self.global_weights = tuple(float(w) for w in new_weights)
        new_head_state = vector_to_state(self.global_weights, self.schema)
        set_state(self.global_model, new_head_state, strict=False)
        return self.global_weights

    def evaluate_global(self, threshold: float = 0.50) -> Dict[str, Any]:
        """Evaluate current global model on validation and test splits."""
        val_loss, val_trues, val_probs = evaluate_model(
            self.global_model, self.val_loader, self.device
        )
        val_m = compute_metrics(val_trues, val_probs, threshold=threshold)

        test_loss, test_trues, test_probs = evaluate_model(
            self.global_model, self.test_loader, self.device
        )
        test_m = compute_metrics(test_trues, test_probs, threshold=threshold)

        return {
            "val_loss": round(val_loss, 4),
            "val_accuracy": val_m["accuracy"],
            "val_f1": val_m["f1"],
            "val_auc": val_m["roc_auc"],
            "test_loss": round(test_loss, 4),
            "test_accuracy": test_m["accuracy"],
            "test_f1": test_m["f1"],
            "test_auc": test_m["roc_auc"],
            "test_confusion_matrix": test_m["confusion_matrix"],
        }


# --------------------------------------------------------------------------- #
# Full Federated Experiment Pipeline                                          #
# --------------------------------------------------------------------------- #
def run_federated_experiment(
    train_samples: Sequence[SampleRecord],
    val_samples: Sequence[SampleRecord],
    test_samples: Sequence[SampleRecord],
    output_dir: Path | str,
    num_clients: int = 5,
    rounds: int = 10,
    local_epochs: int = 1,
    batch_size: int = 32,
    lr: float = 1e-3,
    partition_type: str = "iid",
    alpha: float = 0.5,
    mu: float = 0.0,
    seed: int = 42,
    device: Optional[torch.device] = None,
    custom_aggregator: Optional[Callable[[Sequence[ClientUpdate]], Tuple[float, ...]]] = None,
    malicious_fraction: float = 0.0,
    attack_type: str = "none",
    defense: str = "none",
    dp_clip_norm: float | None = None,
    dp_noise_multiplier: float = 0.0,
) -> Dict[str, Any]:
    """Execute complete multi-round federated training experiment with logging."""
    t0 = time.time()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rounds_log_file = out / "rounds.jsonl"

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {
        "num_clients": num_clients,
        "rounds": rounds,
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "partition_type": partition_type,
        "alpha": alpha,
        "mu": mu,
        "seed": seed,
        "malicious_fraction": malicious_fraction,
        "attack_type": attack_type,
        "federated_scope": "head",
        "parameter_count": FEDERATION_CONTRACT["head_parameter_count"],
        "differential_privacy": {
            "enabled": dp_clip_norm is not None,
            "clip_norm": dp_clip_norm,
            "noise_multiplier": dp_noise_multiplier,
            "accounting": "not provided; use a privacy accountant before production deployment",
        },
    }

    from .attack import apply_label_flip, apply_update_scaling
    from .robust_torch import get_robust_aggregator

    num_malicious = int(round(malicious_fraction * num_clients))
    if custom_aggregator is None and defense.lower() != "none":
        custom_aggregator = get_robust_aggregator(defense, byzantine_count=max(1, num_malicious))

    config["defense"] = defense
    config["num_malicious"] = num_malicious

    # 1. Shard training data across clients
    client_shards = partition_dataset(
        train_samples,
        num_clients=num_clients,
        partition_type=partition_type,
        alpha=alpha,
        seed=seed,
    )

    # 2. Instantiate server and clients, injecting adversarial data for malicious clients
    server = FederatedServer(val_samples, test_samples, device=device)
    clients: List[FederatedClient] = []
    malicious_flags: Dict[str, bool] = {}

    for idx, (cid, shard) in enumerate(client_shards.items()):
        is_malicious = idx < num_malicious
        malicious_flags[cid] = is_malicious

        if is_malicious and attack_type == "label_flip":
            poisoned_shard = apply_label_flip(shard, flip_rate=1.0, seed=seed + idx)
            client = FederatedClient(cid, poisoned_shard, device=device)
        else:
            client = FederatedClient(cid, shard, device=device)
        clients.append(client)

    # Calculate bytes communicated per round: (download global + upload local) * num_clients * 4 bytes
    param_count = FEDERATION_CONTRACT["head_parameter_count"]
    bytes_per_round = 2 * num_clients * param_count * 4

    history: List[Dict[str, Any]] = []

    # Reset/clear log file
    with open(rounds_log_file, "w", encoding="utf-8") as f:
        pass

    for r in range(1, rounds + 1):
        round_t0 = time.time()

        # Step A: Broadcast global weights to each client & train locally
        client_updates: List[FLClientUpdate] = []
        for i, client in enumerate(clients):
            is_mal = malicious_flags[client.client_id]

            # Client training
            update = client.train_local(
                global_weights=server.global_weights,
                epochs=local_epochs,
                lr=lr,
                batch_size=batch_size,
                mu=mu,
                seed=seed + r * 100 + i,
                dp_clip_norm=dp_clip_norm,
                dp_noise_multiplier=dp_noise_multiplier,
            )

            # Apply update-scaling attack if configured
            if is_mal and attack_type == "update_scaling":
                poisoned_params = apply_update_scaling(
                    update.parameters, server.global_weights, scale_factor=5.0
                )
                update = FLClientUpdate(
                    client_id=update.client_id,
                    parameters=poisoned_params,
                    num_examples=update.num_examples,
                    local_loss=update.local_loss,
                    update_norm=round(update.update_norm * 5.0, 4),
                )

            client_updates.append(update)

        # Step B: Server aggregation
        server.aggregate_updates(client_updates, custom_aggregator=custom_aggregator)

        # Step C: Evaluate global metrics
        eval_metrics = server.evaluate_global(threshold=0.50)
        round_duration = time.time() - round_t0

        round_entry = {
            "round": r,
            **eval_metrics,
            "client_losses": {u.client_id: u.local_loss for u in client_updates},
            "client_sample_counts": {u.client_id: u.num_examples for u in client_updates},
            "client_update_norms": {u.client_id: u.update_norm for u in client_updates},
            "mean_update_norm": round(float(np.mean([u.update_norm for u in client_updates])), 4),
            "bytes_communicated": bytes_per_round,
            "round_time_sec": round(round_duration, 2),
        }
        history.append(round_entry)

        # Append to rounds.jsonl
        with open(rounds_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(round_entry) + "\n")

    # Select best validation threshold on final checkpoint
    _, val_trues, val_probs = evaluate_model(
        server.global_model, server.val_loader, device
    )
    best_threshold, val_f1 = select_best_threshold(val_trues, val_probs)

    # Final evaluation on test split using optimal threshold
    _, test_trues, test_probs = evaluate_model(
        server.global_model, server.test_loader, device
    )
    final_test_metrics = compute_metrics(test_trues, test_probs, threshold=best_threshold)

    wall_time = time.time() - t0
    provenance = capture_provenance(seed, config, wall_time)

    # Save artifacts
    torch.save(get_state(server.global_model, scope="full"), out / "checkpoint.pt")

    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    with open(out / "provenance.json", "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)

    summary_metrics = {
        "final_round": rounds,
        "selected_threshold": best_threshold,
        "val_f1_at_threshold": round(val_f1, 4),
        "test_metrics": final_test_metrics,
        "total_bytes_communicated": bytes_per_round * rounds,
        "total_wall_time_sec": round(wall_time, 2),
    }
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    generate_dataset_manifest(
        dataset_name=f"CIFAKE-FL-{partition_type}",
        license_note="CC0-Public-Domain",
        seed=seed,
        image_size=64,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        client_partitions=client_shards,
        output_path=out / "manifest.json",
    )

    return summary_metrics
