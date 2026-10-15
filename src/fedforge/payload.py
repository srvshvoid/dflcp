"""Audit the narrow federated payload boundary.

The smoke protocol sends numeric shared parameters and counts, not raw media
or local personalization state.  The recursive audit errs on the side of
rejecting suspicious keys so a future caller cannot accidentally serialize a
filename or private head tensor into an update.
"""

from __future__ import annotations

from dataclasses import dataclass, is_dataclass, asdict
from collections.abc import Mapping, Sequence
from typing import Any


class PayloadAuditError(ValueError):
    """Raised when an outgoing payload contains forbidden data."""


@dataclass(frozen=True, slots=True)
class PayloadAuditReport:
    """Result of inspecting a payload recursively."""

    valid: bool
    violations: tuple[str, ...]


_FORBIDDEN_KEY_TERMS = (
    "raw_media",
    "raw-media",
    "media_bytes",
    "image_bytes",
    "video_bytes",
    "raw_frame",
    "raw_frames",
    "filename",
    "file_name",
    "file-path",
    "filepath",
    "media_path",
    "local_head",
    "local-head",
    "local_heads",
    "personalized_head",
    "personalized-head",
    "head_tensor",
    "head-tensor",
)
_MEDIA_VALUE_KEYS = {
    "media",
    "image",
    "video",
    "raw_frame",
    "raw_frames",
    "frame_bytes",
    "frames",
}


def _key_text(key: object) -> str:
    return str(key).strip().lower().replace(" ", "_")


def _walk(value: Any, path: str, violations: list[str]) -> None:
    if isinstance(value, (bytes, bytearray, memoryview)):
        violations.append(f"{path}: raw byte media is forbidden")
        return
    if is_dataclass(value):
        _walk(asdict(value), path, violations)
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = _key_text(key)
            child_path = f"{path}.{key_text}" if path else key_text
            if any(term in key_text for term in _FORBIDDEN_KEY_TERMS):
                violations.append(f"{child_path}: forbidden payload field")
            if key_text in _MEDIA_VALUE_KEYS:
                violations.append(f"{child_path}: raw media field")
            if "file" in key_text or key_text in {"path", "filepath"} or key_text.endswith("_path"):
                violations.append(f"{child_path}: filename/path field")
            _walk(child, child_path, violations)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _walk(child, f"{path}[{index}]", violations)


def audit_payload(
    payload: Any,
    *,
    raise_on_error: bool = True,
) -> PayloadAuditReport:
    """Inspect a payload and reject raw media, filenames, and local heads.

    By default a failed audit raises :class:`PayloadAuditError`; callers that
    need a report can pass ``raise_on_error=False``.
    """

    violations: list[str] = []
    _walk(payload, "", violations)
    report = PayloadAuditReport(valid=not violations, violations=tuple(violations))
    if violations and raise_on_error:
        raise PayloadAuditError("unsafe federated payload: " + "; ".join(violations))
    return report


def assert_payload_safe(payload: Any) -> PayloadAuditReport:
    """Explicit assertion alias for :func:`audit_payload`."""

    return audit_payload(payload, raise_on_error=True)


def payload_is_safe(payload: Any) -> bool:
    """Return a boolean without raising for UI/config validation paths."""

    return audit_payload(payload, raise_on_error=False).valid
