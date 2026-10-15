"""Run provenance, JSONL metric, and checkpoint persistence.

The store writes plain JSON files so a smoke run can be replayed offline.  It
captures environment identifiers when available, but records ``unknown``
instead of implying that a git or dependency lock was present.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from typing import Any, Iterable, Mapping


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(child) for child in value]
    if isinstance(value, set):
        return sorted(_jsonable(child) for child in value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git_commit(project_root: Path | None) -> str:
    if project_root is None:
        return "unknown"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    commit = result.stdout.strip()
    return commit or "unknown"


def _lock_hash(project_root: Path | None) -> str:
    if project_root is None:
        return "unknown"
    candidates = (project_root / "requirements.lock", project_root / "poetry.lock")
    for candidate in candidates:
        if candidate.is_file():
            return hashlib.sha256(candidate.read_bytes()).hexdigest()
    return "unknown"


@dataclass(frozen=True, slots=True)
class RunProvenance:
    """Metadata needed to identify how a run was produced."""

    run_id: str
    seed: int
    config: Mapping[str, Any]
    git_commit: str
    python_version: str
    lock_hash: str
    created_at: str

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready provenance mapping."""

        return _jsonable(asdict(self))


def make_provenance(
    *,
    run_id: str,
    seed: int,
    config: Mapping[str, Any],
    project_root: Path | str | None = None,
    created_at: str | None = None,
) -> RunProvenance:
    """Build provenance from a resolved config and local environment."""

    root = Path(project_root) if project_root is not None else None
    timestamp = created_at or datetime.now(timezone.utc).isoformat()
    return RunProvenance(
        run_id=run_id,
        seed=seed,
        config=dict(config),
        git_commit=_git_commit(root),
        python_version=platform.python_version(),
        lock_hash=_lock_hash(root),
        created_at=timestamp,
    )


class RunStore:
    """Persist one run under a directory with stable JSON formatting."""

    def __init__(self, run_dir: Path | str) -> None:
        self.path = Path(run_dir)
        self.path.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.path / "metrics.jsonl"
        self.provenance_path = self.path / "provenance.json"
        self.checkpoint_path = self.path / "checkpoint.json"

    def write_provenance(self, provenance: RunProvenance | Mapping[str, Any]) -> Path:
        """Write or replace ``provenance.json``."""

        _write_json(self.provenance_path, provenance)
        return self.provenance_path

    def append_metric(self, metric: Mapping[str, Any]) -> Path:
        """Append one sorted JSON object to ``metrics.jsonl``."""

        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_jsonable(metric), sort_keys=True) + "\n")
        return self.metrics_path

    def write_metrics(self, metrics: Iterable[Mapping[str, Any]]) -> Path:
        """Replace the JSONL stream with the provided metric records."""

        temporary = self.metrics_path.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for metric in metrics:
                handle.write(json.dumps(_jsonable(metric), sort_keys=True) + "\n")
        temporary.replace(self.metrics_path)
        return self.metrics_path

    def save_checkpoint(self, checkpoint: Mapping[str, Any], *, filename: str = "checkpoint.json") -> Path:
        """Persist a JSON checkpoint and return its path."""

        if not filename or Path(filename).name != filename or not filename.endswith(".json"):
            raise ValueError("checkpoint filename must be a simple .json filename")
        path = self.path / filename
        _write_json(path, checkpoint)
        return path

    def load_checkpoint(self, *, filename: str = "checkpoint.json") -> dict[str, Any]:
        """Read a previously saved checkpoint."""

        path = self.path / filename
        with path.open(encoding="utf-8") as handle:
            result = json.load(handle)
        if not isinstance(result, dict):
            raise ValueError("checkpoint root must be a JSON object")
        return result


def write_provenance(
    run_dir: Path | str,
    provenance: RunProvenance | Mapping[str, Any],
) -> Path:
    """Functional wrapper for :class:`RunStore.write_provenance`."""

    return RunStore(run_dir).write_provenance(provenance)


def append_metric(run_dir: Path | str, metric: Mapping[str, Any]) -> Path:
    """Functional wrapper for :class:`RunStore.append_metric`."""

    return RunStore(run_dir).append_metric(metric)


def save_checkpoint(
    run_dir: Path | str,
    checkpoint: Mapping[str, Any],
    *,
    filename: str = "checkpoint.json",
) -> Path:
    """Functional wrapper for :class:`RunStore.save_checkpoint`."""

    return RunStore(run_dir).save_checkpoint(checkpoint, filename=filename)
