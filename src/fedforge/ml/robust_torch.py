"""Tensor-based robust aggregation primitives for PyTorch, matching fedforge.robust.

Algorithms:
1. Coordinate-wise Median
2. Coordinate-wise Trimmed Mean
3. Multi-client Krum Byzantine selection
4. Global update L2 norm clipping
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import torch

from ..federation import ClientUpdate
from ..robust import (
    coordinate_median as py_coordinate_median,
    krum as py_krum,
    norm_clip as py_norm_clip,
    trimmed_mean as py_trimmed_mean,
)


# --------------------------------------------------------------------------- #
# Tensor-based Primitives                                                     #
# --------------------------------------------------------------------------- #
def torch_coordinate_median(tensors: torch.Tensor) -> torch.Tensor:
    """Compute the coordinate-wise median across client updates (dim 0).

    Args:
        tensors: Tensor of shape (K, D) where K = num_clients, D = num_parameters.

    Returns:
        Tensor of shape (D,) representing median coordinates.
    """
    if tensors.ndim != 2 or tensors.size(0) == 0:
        raise ValueError(f"Expected 2D tensor (K, D) with K >= 1, got shape {tensors.shape}")

    k = tensors.size(0)
    sorted_tensors, _ = torch.sort(tensors, dim=0)
    mid = k // 2
    if k % 2 == 1:
        return sorted_tensors[mid]
    else:
        return (sorted_tensors[mid - 1] + sorted_tensors[mid]) / 2.0


def torch_trimmed_mean(
    tensors: torch.Tensor,
    trim_fraction: float = 0.2,
    trim_count: Optional[int] = None,
) -> torch.Tensor:
    """Compute coordinate-wise trimmed mean, rejecting extreme coordinates."""
    if tensors.ndim != 2 or tensors.size(0) == 0:
        raise ValueError(f"Expected 2D tensor (K, D) with K >= 1, got shape {tensors.shape}")

    k = tensors.size(0)
    if trim_count is None:
        if not (0.0 <= trim_fraction < 0.5):
            raise ValueError("trim_fraction must be in [0.0, 0.5)")
        trim_count = int(math.floor(trim_fraction * k))

    if trim_count < 0 or 2 * trim_count >= k:
        raise ValueError(f"trim_count ({trim_count}) removes all {k} updates.")

    sorted_tensors, _ = torch.sort(tensors, dim=0)
    remaining = sorted_tensors[trim_count: k - trim_count]
    return torch.mean(remaining, dim=0)


def torch_krum(
    tensors: torch.Tensor,
    byzantine_count: int = 1,
) -> torch.Tensor:
    """Select the update minimizing sum of squared Euclidean distances to nearest peers.

    Theoretical guarantee requires K > 2f + 2.
    """
    if tensors.ndim != 2 or tensors.size(0) == 0:
        raise ValueError(f"Expected 2D tensor (K, D), got shape {tensors.shape}")

    k = tensors.size(0)
    f = byzantine_count
    if f < 0:
        raise ValueError("byzantine_count must be non-negative")
    if k <= 2 * f + 2:
        raise ValueError(f"Krum requires n > 2f + 2 updates (n={k}, f={f})")

    neighbor_count = k - f - 2

    # Pairwise squared Euclidean distance matrix (K, K)
    # dist[i, j] = || x_i - x_j ||^2
    diff = tensors.unsqueeze(1) - tensors.unsqueeze(0)  # (K, K, D)
    dist_matrix = torch.sum(diff ** 2, dim=-1)           # (K, K)

    scores = []
    for i in range(k):
        # Exclude self distance (index i)
        other_distances = torch.cat([dist_matrix[i, :i], dist_matrix[i, i + 1:]])
        sorted_dists, _ = torch.sort(other_distances)
        score = torch.sum(sorted_dists[:neighbor_count]).item()
        scores.append(score)

    selected_idx = int(torch.argmin(torch.tensor(scores)).item())
    return tensors[selected_idx]


def torch_norm_clip(
    tensors: torch.Tensor,
    global_tensor: torch.Tensor,
    max_norm: float = 1.0,
) -> torch.Tensor:
    """Scale each update vector towards global weights if its update norm exceeds max_norm."""
    deltas = tensors - global_tensor.unsqueeze(0)  # (K, D)
    norms = torch.norm(deltas, p=2, dim=-1, keepdim=True)  # (K, 1)
    scale = torch.clamp(max_norm / (norms + 1e-8), max=1.0)
    clipped_deltas = deltas * scale
    return global_tensor.unsqueeze(0) + clipped_deltas


# --------------------------------------------------------------------------- #
# Unified Robust Aggregator Factory                                           #
# --------------------------------------------------------------------------- #
def get_robust_aggregator(
    defense_name: str,
    byzantine_count: int = 1,
    max_norm: float = 0.5,
) -> Callable[[Sequence[ClientUpdate]], Tuple[float, ...]]:
    """Return a callable aggregator conforming to the FedForge server interface."""
    def_clean = defense_name.lower().strip()

    if def_clean in ("none", "fedavg"):
        from ..federation import fedavg
        return lambda updates: fedavg(updates)

    elif def_clean == "coordinate_median":
        return lambda updates: py_coordinate_median([u.parameters for u in updates])

    elif def_clean == "trimmed_mean":
        return lambda updates: py_trimmed_mean([u.parameters for u in updates], trim_fraction=0.2)

    elif def_clean == "krum":
        def _krum_agg(updates: Sequence[ClientUpdate]) -> Tuple[float, ...]:
            k = len(updates)
            # Check theoretical feasibility: n > 2f + 2
            f = byzantine_count
            if k <= 2 * f + 2:
                # If f exceeds bound on 5 clients, fallback to f=1 or median
                f = max(0, (k - 3) // 2)
            if f < 1:
                return py_coordinate_median([u.parameters for u in updates])
            return py_krum([u.parameters for u in updates], byzantine_count=f)
        return _krum_agg

    elif def_clean == "norm_clip":
        def _norm_clip_agg(updates: Sequence[ClientUpdate]) -> Tuple[float, ...]:
            result = py_norm_clip([u.parameters for u in updates], max_norm=max_norm, aggregate=True)
            return tuple(result)
        return _norm_clip_agg

    else:
        raise ValueError(
            f"Unknown defense '{defense_name}'. Allowed: 'none', 'coordinate_median', 'trimmed_mean', 'krum', 'norm_clip'."
        )
