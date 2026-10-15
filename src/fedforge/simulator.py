"""Command-line and Python entry point for the deterministic CPU smoke run.

The simulator joins the core contracts without external dependencies: it
generates synthetic videos, creates source-disjoint clients, trains a small
logistic model, optionally applies a semantic label flip, evaluates video
means, and writes provenance/metrics/checkpoint JSON files.  Results are a
reproducibility smoke signal, not real-data research evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .attack import LabelFlipAttack
from .data import (
    SyntheticConfig,
    VideoRecord,
    assert_client_partition_no_leakage,
    generate_synthetic_videos,
    partition_clients,
    split_train_test,
)
from .federation import ClientUpdate, fedavg, personalized_aggregate, train_client_update
from .metrics import accuracy, auc, ece, f1, per_generator_recall
from .model import LogisticFrameModel
from .payload import audit_payload
from .persistence import RunProvenance, RunStore, make_provenance
from .robust import aggregate_vectors, validate_secagg_compatibility


@dataclass(frozen=True, slots=True)
class SimulatorConfig:
    """Resolved options for a deterministic synthetic experiment."""

    seed: int = 11
    n_videos: int = 24
    frames_per_video: int = 4
    n_features: int = 4
    source_count: int = 8
    n_clients: int = 4
    rounds: int = 3
    local_epochs: int = 2
    learning_rate: float = 0.25
    algorithm: str = "fedavg"
    aggregator: str = "fedavg"
    trim_fraction: float = 0.2
    byzantine_count: int = 1
    max_norm: float = 1.0
    secagg_enabled: bool = False
    personalized_local_head: bool = False
    test_fraction: float = 0.25
    attack_generator: str | None = None
    malicious_clients: tuple[str, ...] = ("client-0",)

    def __post_init__(self) -> None:
        if self.rounds < 1 or self.local_epochs < 1:
            raise ValueError("rounds and local_epochs must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.n_clients < 1:
            raise ValueError("n_clients must be positive")
        if not 0.0 < self.test_fraction < 1.0:
            raise ValueError("test_fraction must be between zero and one")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SimulatorConfig":
        """Resolve JSON config keys while accepting common readable aliases."""

        aliases = {
            "secagg": "secagg_enabled",
            "personalized": "personalized_local_head",
            "personalized_head": "personalized_local_head",
            "robust_aggregator": "aggregator",
            "clients": "n_clients",
        }
        normalized: dict[str, Any] = {}
        field_names = set(cls.__dataclass_fields__)
        for key, value in values.items():
            target = aliases.get(key, key)
            if target in field_names:
                normalized[target] = value
        if "malicious_clients" in normalized:
            normalized["malicious_clients"] = tuple(normalized["malicious_clients"])
        return cls(**normalized)

    def resolved_dict(self) -> dict[str, Any]:
        """Return a JSON-ready resolved configuration."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Paths and final metrics returned by :func:`run_smoke_experiment`."""

    run_dir: Path
    metrics: tuple[dict[str, Any], ...]
    provenance: RunProvenance
    checkpoint: Path


def load_config(config_path: Path | str) -> SimulatorConfig:
    """Load and resolve a JSON simulator config."""

    path = Path(config_path)
    with path.open(encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, dict):
        raise ValueError("simulator config must be a JSON object")
    return SimulatorConfig.from_mapping(values)


def _run_id(config: SimulatorConfig) -> str:
    encoded = json.dumps(config.resolved_dict(), sort_keys=True, separators=(",", ":")).encode()
    return "smoke-" + hashlib.sha256(encoded).hexdigest()[:12]


def _attack_clients(
    clients: Mapping[str, Sequence[VideoRecord]],
    config: SimulatorConfig,
) -> dict[str, tuple[VideoRecord, ...]]:
    if not config.attack_generator:
        return {client_id: tuple(videos) for client_id, videos in clients.items()}
    attack = LabelFlipAttack(target_generator=config.attack_generator)
    return attack.apply_to_clients(
        clients,
        malicious_client_ids=config.malicious_clients,
    )


def _evaluate(
    model: LogisticFrameModel,
    videos: Sequence[VideoRecord],
) -> dict[str, Any]:
    labels = [video.label for video in videos]
    probabilities = [model.predict_video_proba(video) for video in videos]
    predictions = [int(probability >= 0.5) for probability in probabilities]
    return {
        "accuracy": accuracy(labels, predictions),
        "f1": f1(labels, predictions),
        "auc": auc(labels, probabilities),
        "ece": ece(labels, probabilities),
        "per_generator_recall": per_generator_recall(videos, predictions),
        "video_count": len(videos),
    }


def run_smoke_experiment(
    config_path: Path | str,
    run_dir: Path | str | None = None,
) -> SimulationResult:
    """Run a deterministic CPU experiment described by a JSON config path."""

    config = load_config(config_path)
    validate_secagg_compatibility(config.secagg_enabled, config.aggregator)
    root = Path(__file__).resolve().parents[2]
    resolved_run_dir = (
        Path(run_dir)
        if run_dir is not None
        else root / "runs" / _run_id(config)
    )
    store = RunStore(resolved_run_dir)
    provenance = make_provenance(
        run_id=_run_id(config),
        seed=config.seed,
        config=config.resolved_dict(),
        project_root=root,
    )
    store.write_provenance(provenance)

    videos = generate_synthetic_videos(
        SyntheticConfig(
            seed=config.seed,
            n_videos=config.n_videos,
            frames_per_video=config.frames_per_video,
            n_features=config.n_features,
            source_count=config.source_count,
        )
    )
    train_videos, test_videos = split_train_test(
        videos,
        test_fraction=config.test_fraction,
        seed=config.seed,
        source_disjoint=True,
    )
    clients = partition_clients(
        train_videos,
        config.n_clients,
        seed=config.seed,
        source_disjoint=True,
    )
    assert_client_partition_no_leakage(clients, require_source_disjoint=True)
    attacked_clients = _attack_clients(clients, config)

    model = LogisticFrameModel.zeros(config.n_features)
    local_heads: dict[str, float] = {client_id: 0.0 for client_id in clients}
    metric_rows: list[dict[str, Any]] = []
    for round_index in range(1, config.rounds + 1):
        updates: list[ClientUpdate] = []
        for client_id in sorted(attacked_clients):
            client_videos = attacked_clients[client_id]
            if not client_videos:
                continue
            update = train_client_update(
                model,
                client_videos,
                client_id=client_id,
                epochs=config.local_epochs,
                learning_rate=config.learning_rate,
                algorithm=config.algorithm,
                proximal_mu=0.1 if config.algorithm.lower() == "fedprox" else 0.0,
                local_head=local_heads.get(client_id, 0.0),
                personalize_head=config.personalized_local_head,
            )
            # Audit the actual outgoing shape before any server-side operation.
            audit_payload(update.to_payload())
            updates.append(update)
        if not updates:
            raise ValueError("no non-empty client updates were produced")
        if config.aggregator.lower().replace("-", "_") in {"fedavg", "fedprox"}:
            model = LogisticFrameModel.from_parameters(fedavg(updates))
        else:
            model = LogisticFrameModel.from_parameters(
                aggregate_vectors(
                    [update.parameters for update in updates],
                    aggregator=config.aggregator,
                    trim_fraction=config.trim_fraction,
                    byzantine_count=config.byzantine_count,
                    max_norm=config.max_norm,
                    secagg_enabled=config.secagg_enabled,
                )
            )
        if config.personalized_local_head:
            state = personalized_aggregate(updates)
            local_heads.update(
                {client_id: values[0] for client_id, values in state.local_heads.items()}
            )
        evaluation = _evaluate(model, test_videos)
        row = {"round": round_index, **evaluation}
        metric_rows.append(row)
    store.write_metrics(metric_rows)

    checkpoint = store.save_checkpoint(
        {
            "model_parameters": list(model.parameters()),
            "local_heads": dict(sorted(local_heads.items())) if config.personalized_local_head else {},
            "config": config.resolved_dict(),
            "final_metrics": metric_rows[-1],
        }
    )
    return SimulationResult(
        run_dir=resolved_run_dir,
        metrics=tuple(metric_rows),
        provenance=provenance,
        checkpoint=checkpoint,
    )


def run_from_config(
    config_path: Path | str,
    run_dir: Path | str | None = None,
) -> SimulationResult:
    """Alias emphasizing that the entry point consumes a config path."""

    return run_smoke_experiment(config_path, run_dir)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI for ``python -m fedforge.simulator``."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_positional", nargs="?", help="JSON config path")
    parser.add_argument("output_positional", nargs="?", help="run directory")
    parser.add_argument("--config", dest="config_flag", help="JSON config path")
    parser.add_argument("--output", dest="output_flag", help="run directory")
    args = parser.parse_args(argv)
    config_path = args.config_flag or args.config_positional
    if not config_path:
        parser.error("a config path is required")
    output = args.output_flag or args.output_positional
    result = run_smoke_experiment(config_path, output)
    print(str(result.run_dir))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the CLI smoke path
    raise SystemExit(main())
