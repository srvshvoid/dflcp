"""Public aggregation imports for small experiment scripts."""

from .federation import (
    ClientUpdate,
    FederatedState,
    fedavg,
    fedprox,
    fedprox_step,
    personalized_aggregate,
)
from .robust import (
    CompatibilityError,
    aggregate_vectors,
    clip_by_l2_norm,
    coordinate_median,
    krum,
    norm_clip,
    trimmed_mean,
    validate_secagg_compatibility,
)

__all__ = [
    "ClientUpdate",
    "CompatibilityError",
    "FederatedState",
    "aggregate_vectors",
    "clip_by_l2_norm",
    "coordinate_median",
    "fedavg",
    "fedprox",
    "fedprox_step",
    "krum",
    "norm_clip",
    "personalized_aggregate",
    "trimmed_mean",
    "validate_secagg_compatibility",
]
