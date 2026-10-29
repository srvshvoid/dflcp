"""Adversarial threat models for federated learning in FedForge.

Implements:
1. Data Poisoning: Label-flipping attack (symmetrical and targeted deepfake evasion).
2. Model Poisoning: Parameter update scaling and sign-inversion attack.
"""

from __future__ import annotations

import copy
from typing import List, Optional, Sequence, Tuple
import numpy as np

from .data import INV_LABEL_MAP, SampleRecord


def apply_label_flip(
    samples: Sequence[SampleRecord],
    flip_rate: float = 1.0,
    targeted_class: Optional[int] = None,
    seed: int = 42,
) -> List[SampleRecord]:
    """Apply label-flipping poisoning to a client's local dataset.

    Args:
        samples: Original sample records.
        flip_rate: Fraction of candidate samples to flip (0.0 to 1.0).
        targeted_class:
            If None: Symmetrical flip: 0 -> 1 and 1 -> 0.
            If 1: Targeted evasion: 1 (fake) -> 0 (real), teaching model that deepfakes are authentic.
            If 0: Denial-of-service: 0 (real) -> 1 (fake), causing false alarms on authentic photos.
        seed: Random seed for reproducibility.

    Returns:
        List of poisoned SampleRecord objects.
    """
    rng = np.random.default_rng(seed)
    poisoned: List[SampleRecord] = []

    for s in samples:
        should_flip = rng.uniform(0.0, 1.0) < flip_rate
        new_label = s.label

        if should_flip:
            if targeted_class is None:
                # Symmetrical flip
                new_label = 1 - s.label
            elif s.label == targeted_class:
                # Targeted flip
                new_label = 1 - targeted_class

        poisoned.append(
            SampleRecord(
                filepath=s.filepath,
                label=new_label,
                sha256=s.sha256,
                class_name=INV_LABEL_MAP[new_label],
            )
        )

    return poisoned


def apply_update_scaling(
    local_weights: Sequence[float],
    global_weights: Sequence[float],
    scale_factor: float = 5.0,
) -> Tuple[float, ...]:
    """Scale the parameter update Delta w by a multiplier to steer aggregation.

    w_poisoned = w_global + scale_factor * (w_local - w_global)
    """
    poisoned = [
        g + scale_factor * (l - g) for l, g in zip(local_weights, global_weights)
    ]
    return tuple(poisoned)
