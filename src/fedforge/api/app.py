"""Read-only run API and optional FastAPI adapter.

The repository implementation is dependency-free and deliberately exposes
only run metadata, JSONL metrics, and an explicitly labelled numeric smoke
inference endpoint.  Image/video multipart inference, authentication,
PostgreSQL, Redis, and live WebSocket events remain pending integrations; the
``/detect`` route says so instead of pretending to score uploaded media.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..model import LogisticFrameModel


class RunNotFoundError(FileNotFoundError):
    """Raised when a requested run is not present below the configured root."""


@dataclass(frozen=True, slots=True)
class RunRepository:
    """Read smoke run artifacts without a database or network connection."""

    runs_root: Path

    def __init__(self, runs_root: Path | str = "runs") -> None:
        object.__setattr__(self, "runs_root", Path(runs_root))

    def _run_path(self, run_id: str) -> Path:
        """Resolve a run id and reject path traversal."""

        if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
            raise RunNotFoundError(run_id)
        path = self.runs_root / run_id
        if path.is_dir():
            return path
        # Allow callers to use the provenance run_id returned by older replay
        # artifacts while keeping path traversal impossible.
        if self.runs_root.is_dir():
            for candidate in self.runs_root.iterdir():
                provenance_path = candidate / "provenance.json"
                if not candidate.is_dir() or not provenance_path.is_file():
                    continue
                try:
                    provenance = self._read_json(provenance_path)
                except (OSError, ValueError, json.JSONDecodeError):
                    continue
                if provenance.get("run_id") == run_id:
                    return candidate
        raise RunNotFoundError(run_id)

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError(f"{path} must contain a JSON object")
        return value

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} is not a JSON object")
                rows.append(value)
        return rows

    def list_runs(self) -> list[dict[str, Any]]:
        """Return safe summaries for every run with provenance."""

        if not self.runs_root.is_dir():
            return []
        summaries: list[dict[str, Any]] = []
        for path in sorted(self.runs_root.iterdir(), key=lambda item: item.name):
            provenance_path = path / "provenance.json"
            if not path.is_dir() or not provenance_path.is_file():
                continue
            provenance = self._read_json(provenance_path)
            metrics = self._read_jsonl(path / "metrics.jsonl")
            summaries.append(
                {
                    # The directory is the stable URL key. A caller can still
                    # inspect the run's internal provenance id separately.
                    "id": path.name,
                    "run_id": provenance.get("run_id", path.name),
                    "directory": path.name,
                    "seed": provenance.get("seed"),
                    "git_commit": provenance.get("git_commit"),
                    "lock_hash": provenance.get("lock_hash"),
                    "created_at": provenance.get("created_at"),
                    "status": "replay",
                    "metrics_count": len(metrics),
                    "evidence_status": "synthetic-unverified",
                }
            )
        return summaries

    def get_run(self, run_id: str) -> dict[str, Any]:
        """Return provenance and metrics for one run."""

        path = self._run_path(run_id)
        provenance = self._read_json(path / "provenance.json")
        return {
            "id": run_id,
            "provenance": provenance,
            "metrics": self._read_jsonl(path / "metrics.jsonl"),
            "checkpoint_available": (path / "checkpoint.json").is_file(),
            "evidence_status": "synthetic-unverified",
        }

    def rounds(self, run_id: str) -> list[dict[str, Any]]:
        """Return only the JSONL round records for one run."""

        path = self._run_path(run_id)
        return self._read_jsonl(path / "metrics.jsonl")

    def privacy_payload_manifest(self) -> dict[str, Any]:
        """Describe the smoke payload audit without overstating measurement."""

        return {
            "mode": "offline-smoke",
            "measured_transport": False,
            "status": "schema-and-audit-only",
            "allowed_fields": ["client_id", "parameters", "num_examples", "metadata"],
            "forbidden_fields": [
                "raw_media",
                "filenames",
                "media_paths",
                "local_head",
                "personalized_head",
            ],
            "note": (
                "The simulator audits an outgoing payload shape but does not run "
                "a cryptographic transport or persist measured network bytes."
            ),
        }


def _safe_checkpoint_path(repository: RunRepository, checkpoint: str | None) -> Path:
    """Resolve a checkpoint relative to the configured runs root."""

    if checkpoint is None:
        runs = repository.list_runs()
        if not runs:
            raise RunNotFoundError("no runs available")
        checkpoint = str(runs[-1]["directory"])
    run_path = repository._run_path(checkpoint)
    path = run_path / "checkpoint.json"
    if not path.is_file():
        raise RunNotFoundError(checkpoint)
    return path


def score_smoke_features(
    features: Sequence[float],
    *,
    checkpoint: Path | str,
    local_head: float = 0.0,
) -> dict[str, Any]:
    """Score one numeric smoke feature vector from a saved checkpoint.

    This endpoint is intentionally not image/video inference.  It exists to
    make the checkpoint boundary testable before the optional media stack is
    added and marks calibration/explanations as unavailable.
    """

    checkpoint_path = Path(checkpoint)
    with checkpoint_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model_parameters"), list):
        raise ValueError("checkpoint has no model_parameters list")
    model = LogisticFrameModel.from_parameters(payload["model_parameters"])
    probability = model.predict_proba(features, local_head=local_head)
    return {
        "mode": "numeric-smoke",
        "score": probability,
        "label": "fake" if probability >= 0.5 else "real",
        "calibrated": False,
        "calibration_note": "Temperature scaling is pending real validation data.",
        "heatmap": None,
        "face_crops": [],
        "frame_scores": [probability],
        "latency_ms": None,
    }


def create_fastapi_app(runs_root: Path | str = "runs") -> Any:
    """Create the optional FastAPI adapter.

    FastAPI is intentionally an optional deployment dependency.  Importing
    ``fedforge.api`` remains possible in the offline smoke environment; this
    factory gives a targeted error if a caller requests the service without
    installing the separately pinned API stack.
    """

    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import JSONResponse
    except ImportError as exc:  # pragma: no cover - depends on optional stack
        raise RuntimeError(
            "FastAPI is not installed; use the offline RunRepository or install "
            "the separately reviewed API dependencies."
        ) from exc

    repository = RunRepository(runs_root)
    app = FastAPI(title="FedForge API", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "mode": "offline-smoke", "live_services": False}

    @app.get("/runs")
    def runs() -> list[dict[str, Any]]:
        return repository.list_runs()

    @app.get("/runs/{run_id}")
    def run(run_id: str) -> dict[str, Any]:
        try:
            return repository.get_run(run_id)
        except (RunNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/runs/{run_id}/rounds")
    def rounds(run_id: str) -> list[dict[str, Any]]:
        try:
            return repository.rounds(run_id)
        except (RunNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/privacy/payload-manifest")
    def payload_manifest() -> dict[str, Any]:
        return repository.privacy_payload_manifest()

    @app.post("/detect")
    async def detect(request: Request) -> JSONResponse:
        # Do not silently accept an upload and return a score from the numeric
        # fixture.  A real implementation needs file sniffing, sandboxed media
        # decode, face extraction, calibration, and retention controls.
        return JSONResponse(
            status_code=501,
            content={
                "detail": (
                    "Image/video/audio inference is pending the optional media "
                    "stack; no uploaded bytes were processed."
                ),
                "mode": "offline-smoke",
            },
        )

    @app.post("/detect/smoke")
    async def detect_smoke(request: Request) -> dict[str, Any]:
        body = await request.json()
        if not isinstance(body, Mapping):
            raise HTTPException(status_code=400, detail="JSON object required")
        raw_features = body.get("features")
        if not isinstance(raw_features, list) or not raw_features:
            raise HTTPException(status_code=400, detail="features must be a non-empty list")
        try:
            features = [float(value) for value in raw_features]
            checkpoint_path = _safe_checkpoint_path(repository, body.get("checkpoint"))
            result = score_smoke_features(
                features,
                checkpoint=checkpoint_path,
                local_head=float(body.get("local_head", 0.0)),
            )
        except (TypeError, ValueError, OSError, RunNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return result

    return app


__all__ = ["RunNotFoundError", "RunRepository", "create_fastapi_app", "score_smoke_features"]
