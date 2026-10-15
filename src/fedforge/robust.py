"""Plaintext robust aggregation primitives for toy federated updates.

These functions operate on small numeric vectors and are intended for smoke
experiments and hand-checkable tests.  They do not claim to provide a secure
or production-hardened Byzantine-robust protocol.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence


class CompatibilityError(ValueError):
    """Raised for a SecAgg/aggregation combination this slice cannot support."""


Vector = tuple[float, ...]


def _coerce_vectors(vectors: Iterable[Sequence[float]]) -> list[Vector]:
    result = [tuple(float(value) for value in vector) for vector in vectors]
    if not result or not result[0]:
        raise ValueError("at least one non-empty update is required")
    dimension = len(result[0])
    if any(len(vector) != dimension for vector in result):
        raise ValueError("all updates must have equal dimensions")
    return result


def _quantile_median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def trimmed_mean(
    vectors: Iterable[Sequence[float]],
    *,
    trim_fraction: float = 0.2,
    trim_count: int | None = None,
) -> Vector:
    """Return a coordinate-wise trimmed mean.

    ``trim_count`` removes that many low and high values per coordinate.  If
    omitted, the count is ``floor(trim_fraction * n)``.  The input must retain
    at least one value after both tails are removed.
    """

    items = _coerce_vectors(vectors)
    if trim_count is None:
        if not 0.0 <= trim_fraction < 0.5:
            raise ValueError("trim_fraction must be in [0, 0.5)")
        trim_count = int(math.floor(trim_fraction * len(items)))
    if trim_count < 0 or 2 * trim_count >= len(items):
        raise ValueError("trim_count removes all available updates")
    return tuple(
        sum(sorted(vector[index] for vector in items)[trim_count : len(items) - trim_count])
        / (len(items) - 2 * trim_count)
        for index in range(len(items[0]))
    )


def coordinate_median(vectors: Iterable[Sequence[float]]) -> Vector:
    """Return the median independently in each parameter coordinate."""

    items = _coerce_vectors(vectors)
    return tuple(
        _quantile_median([vector[index] for vector in items])
        for index in range(len(items[0]))
    )


def _squared_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right))


def krum(
    vectors: Iterable[Sequence[float]],
    *,
    byzantine_count: int = 1,
    f: int | None = None,
) -> Vector:
    """Return the Krum-selected update for a small plaintext cohort.

    The classic Krum score sums distances to the closest ``n - f - 2``
    peers, and the first minimum is selected deterministically.  The usual
    requirement ``n > 2f + 2`` is enforced.  This is a numeric primitive; it
    does not identify whether a real client is Byzantine.
    """

    items = _coerce_vectors(vectors)
    fault_count = byzantine_count if f is None else f
    if fault_count < 0:
        raise ValueError("byzantine_count must be non-negative")
    if len(items) <= 2 * fault_count + 2:
        raise ValueError("Krum requires n > 2f + 2 updates")
    neighbor_count = len(items) - fault_count - 2
    scores: list[float] = []
    for index, candidate in enumerate(items):
        distances = sorted(
            _squared_distance(candidate, other)
            for other_index, other in enumerate(items)
            if other_index != index
        )
        scores.append(sum(distances[:neighbor_count]))
    selected_index = min(range(len(items)), key=lambda index: (scores[index], index))
    return items[selected_index]


def l2_norm(vector: Sequence[float]) -> float:
    """Return a vector's Euclidean norm."""

    return math.sqrt(sum(float(value) ** 2 for value in vector))


def clip_by_l2_norm(vector: Sequence[float], max_norm: float) -> Vector:
    """Scale one vector down to ``max_norm`` when it exceeds the bound."""

    if max_norm <= 0.0:
        raise ValueError("max_norm must be positive")
    values = tuple(float(value) for value in vector)
    norm = l2_norm(values)
    if norm <= max_norm or norm == 0.0:
        return values
    scale = max_norm / norm
    return tuple(value * scale for value in values)


def norm_clip(
    vectors: Iterable[Sequence[float]] | Sequence[float],
    max_norm: float,
    *,
    aggregate: bool = True,
) -> Vector | tuple[Vector, ...]:
    """Clip updates by norm and optionally return their mean.

    Client-side clipping can be performed before additive SecAgg.  The
    default aggregate form represents the server's plaintext clipped mean;
    callers using SecAgg should use ``aggregate=False`` and then add the
    clipped vectors through their protocol.
    """

    raw_values = list(vectors)
    if raw_values and all(isinstance(value, (int, float)) for value in raw_values):
        # A single-update convenience form is useful at a client boundary.
        return clip_by_l2_norm(raw_values, max_norm)
    items = _coerce_vectors(raw_values)  # type: ignore[arg-type]
    clipped = tuple(clip_by_l2_norm(vector, max_norm) for vector in items)
    if not aggregate:
        return clipped
    return tuple(
        sum(vector[index] for vector in clipped) / len(clipped)
        for index in range(len(clipped[0]))
    )


def norm_clipping(
    vectors: Iterable[Sequence[float]] | Sequence[float],
    max_norm: float,
) -> Vector | tuple[Vector, ...]:
    """Clip one vector or return the plaintext clipped mean for many."""

    result = norm_clip(vectors, max_norm, aggregate=True)
    return result


# Explicit names for callers that want to make the client-side operation
# distinct from the server-side clipped mean.
clip_norm = clip_by_l2_norm
clip_update = clip_by_l2_norm


_ROBUST_PLAINTEXT = frozenset(
    {"trimmed_mean", "trimmed-mean", "coordinate_median", "coordinate-median", "median", "krum", "norm_clip", "norm-clipping", "norm_clipping"}
)
_ADDITIVE = frozenset({"fedavg", "fed-avg", "fedprox", "fed-prox", "mean", "none"})


def validate_secagg_compatibility(
    secagg_enabled: bool,
    aggregator: str,
    *,
    clipping_mode: str = "server",
) -> bool:
    """Validate whether an aggregator fits the available SecAgg boundary.

    FedAvg/FedProx-style additive aggregation is accepted with SecAgg.  The
    robust methods here inspect individual plaintext updates and therefore
    raise :class:`CompatibilityError` when SecAgg is enabled.  Norm clipping
    is accepted only when explicitly declared client-side, where each client
    clips before encryption.
    """

    normalized = aggregator.lower().replace(" ", "_")
    if not secagg_enabled:
        return True
    if normalized in {"norm_clip", "norm_clipping", "norm-clipping", "norm_clip_mean"}:
        if clipping_mode.lower() == "client":
            return True
        raise CompatibilityError(
            "server-side norm clipping needs plaintext updates; use clipping_mode='client' with SecAgg"
        )
    if normalized in _ROBUST_PLAINTEXT:
        raise CompatibilityError(
            f"{aggregator} requires individual plaintext updates and is incompatible with SecAgg"
        )
    if normalized not in _ADDITIVE:
        raise CompatibilityError(f"unknown aggregator: {aggregator}")
    return True


def validate_robust_aggregation_compatibility(
    *,
    secagg_enabled: bool,
    aggregator: str,
    clipping_mode: str = "server",
) -> bool:
    """Keyword-oriented alias for :func:`validate_secagg_compatibility`."""

    return validate_secagg_compatibility(
        secagg_enabled,
        aggregator,
        clipping_mode=clipping_mode,
    )


def aggregate_vectors(
    vectors: Iterable[Sequence[float]],
    *,
    aggregator: str = "fedavg",
    trim_fraction: float = 0.2,
    byzantine_count: int = 1,
    max_norm: float = 1.0,
    secagg_enabled: bool = False,
) -> Vector:
    """Dispatch a numeric aggregation method after compatibility validation."""

    validate_secagg_compatibility(secagg_enabled, aggregator)
    normalized = aggregator.lower().replace("-", "_").replace(" ", "_")
    if normalized in {"fedavg", "fedprox", "mean", "none"}:
        items = _coerce_vectors(vectors)
        return tuple(sum(vector[index] for vector in items) / len(items) for index in range(len(items[0])))
    if normalized == "trimmed_mean":
        return trimmed_mean(vectors, trim_fraction=trim_fraction)
    if normalized in {"coordinate_median", "median"}:
        return coordinate_median(vectors)
    if normalized == "krum":
        return krum(vectors, byzantine_count=byzantine_count)
    if normalized in {"norm_clip", "norm_clipping", "norm_clipping_mean"}:
        return norm_clipping(vectors, max_norm)
    raise ValueError(f"unknown aggregator: {aggregator}")


trimmed_mean_aggregate = trimmed_mean
coordinate_median_aggregate = coordinate_median
krum_aggregate = krum
validate_secagg = validate_secagg_compatibility
