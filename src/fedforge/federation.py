"""Small federated update contracts for the CPU simulator.

The implementation keeps the privacy boundary visible: an update contains a
shared parameter vector and sample count, while a personalized local head is
optional state that is retained by the client and excluded from aggregation.
This is a local educational simulator, not a claim to implement Flower's
SecAgg+ protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from .data import FrameRecord, VideoRecord
from .model import LogisticFrameModel


def _validate_vector(values: Sequence[float], *, name: str = "vector") -> tuple[float, ...]:
    if not values:
        raise ValueError(f"{name} must not be empty")
    return tuple(float(value) for value in values)


@dataclass(frozen=True, slots=True)
class ClientUpdate:
    """One client's shared update and optional private local head.

    ``local_head`` is represented as a tuple to make accidental tensor
    sharing obvious in tests.  It is deliberately omitted from
    :meth:`to_payload` by default; passing ``include_local_head=True`` creates
    an intentionally unsafe payload that the audit module will reject.
    """

    client_id: str
    parameters: tuple[float, ...]
    num_examples: int
    local_head: tuple[float, ...] | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.client_id:
            raise ValueError("client_id is required")
        if not self.parameters:
            raise ValueError("parameters must not be empty")
        if self.num_examples < 1:
            raise ValueError("num_examples must be positive")
        if self.local_head is not None and not self.local_head:
            raise ValueError("local_head must be non-empty when supplied")

    @property
    def shared_parameters(self) -> tuple[float, ...]:
        """Alias emphasizing which values are safe to aggregate."""

        return self.parameters

    def to_payload(self, *, include_local_head: bool = False) -> dict[str, object]:
        """Serialize the update boundary without raw records or filenames."""

        payload: dict[str, object] = {
            "client_id": self.client_id,
            "parameters": list(self.parameters),
            "num_examples": self.num_examples,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        if include_local_head and self.local_head is not None:
            payload["local_head"] = list(self.local_head)
        return payload


@dataclass(frozen=True, slots=True)
class FederatedState:
    """Shared model parameters plus per-client private head state."""

    shared_parameters: tuple[float, ...]
    local_heads: Mapping[str, tuple[float, ...]] = field(default_factory=dict)


def _updates_and_weights(
    updates: Iterable[ClientUpdate | Sequence[float]],
    weights: Sequence[float] | None = None,
) -> tuple[list[tuple[float, ...]], list[float]]:
    items = list(updates)
    if not items:
        raise ValueError("at least one update is required")
    vectors: list[tuple[float, ...]] = []
    default_weights: list[float] = []
    for item in items:
        if isinstance(item, ClientUpdate):
            vectors.append(item.parameters)
            default_weights.append(float(item.num_examples))
        else:
            vectors.append(_validate_vector(item, name="update"))
            default_weights.append(1.0)
    dimension = len(vectors[0])
    if any(len(vector) != dimension for vector in vectors):
        raise ValueError("all updates must have the same dimension")
    final_weights = list(weights) if weights is not None else default_weights
    if len(final_weights) != len(vectors):
        raise ValueError("weights must match the number of updates")
    if any(weight < 0.0 for weight in final_weights):
        raise ValueError("weights must be non-negative")
    if sum(final_weights) <= 0.0:
        raise ValueError("at least one aggregation weight must be positive")
    return vectors, [float(weight) for weight in final_weights]


def fedavg(
    updates: Iterable[ClientUpdate | Sequence[float]],
    *,
    weights: Sequence[float] | None = None,
) -> tuple[float, ...]:
    """Return a sample-count-weighted FedAvg shared parameter vector."""

    vectors, final_weights = _updates_and_weights(updates, weights)
    denominator = sum(final_weights)
    return tuple(
        sum(weight * vector[index] for vector, weight in zip(vectors, final_weights))
        / denominator
        for index in range(len(vectors[0]))
    )


def fedprox_step(
    global_parameters: Sequence[float],
    local_parameters: Sequence[float],
    *,
    learning_rate: float = 1.0,
    proximal_mu: float = 0.0,
) -> tuple[float, ...]:
    """Apply one proximal correction to local parameters.

    If ``local_parameters`` already contains the result of local gradient
    descent, this returns ``local - learning_rate * mu * (local - global)``.
    The full simulator uses the equivalent proximal gradient term during each
    local batch update.
    """

    global_vector = _validate_vector(global_parameters, name="global_parameters")
    local_vector = _validate_vector(local_parameters, name="local_parameters")
    if len(global_vector) != len(local_vector):
        raise ValueError("global and local parameter dimensions must match")
    if learning_rate < 0.0 or proximal_mu < 0.0:
        raise ValueError("learning_rate and proximal_mu must be non-negative")
    return tuple(
        local_value - learning_rate * proximal_mu * (local_value - global_value)
        for global_value, local_value in zip(global_vector, local_vector)
    )


def personalized_aggregate(
    updates: Iterable[ClientUpdate],
    *,
    weights: Sequence[float] | None = None,
) -> FederatedState:
    """Aggregate shared tensors while retaining each client's local head.

    No local head is averaged or placed into ``shared_parameters``.  Missing
    heads are simply absent from the returned private-state mapping.
    """

    update_list = list(updates)
    shared = fedavg(update_list, weights=weights)
    heads = {
        update.client_id: tuple(update.local_head)
        for update in update_list
        if update.local_head is not None
    }
    return FederatedState(shared_parameters=shared, local_heads=heads)


def train_client_update(
    global_model: LogisticFrameModel,
    videos: Iterable[VideoRecord],
    *,
    client_id: str,
    epochs: int = 1,
    learning_rate: float = 0.1,
    algorithm: str = "fedavg",
    proximal_mu: float = 0.0,
    local_head: float = 0.0,
    personalize_head: bool = False,
) -> ClientUpdate:
    """Train one deterministic client update over all of its frames."""

    video_list = tuple(videos)
    frames: tuple[FrameRecord, ...] = tuple(
        frame for video in video_list for frame in video.frames
    )
    if not frames:
        raise ValueError("a client needs at least one frame")
    normalized_algorithm = algorithm.lower().replace("_", "-")
    if normalized_algorithm not in {"fedavg", "fedprox"}:
        raise ValueError(f"unsupported local algorithm: {algorithm}")
    if normalized_algorithm == "fedavg":
        proximal_mu = 0.0
    updated_model, updated_head, _ = global_model.fit_frames(
        frames,
        epochs=epochs,
        learning_rate=learning_rate,
        local_head=local_head,
        personalize_head=personalize_head,
        proximal_reference=global_model.parameters(),
        proximal_mu=proximal_mu,
    )
    return ClientUpdate(
        client_id=client_id,
        parameters=updated_model.parameters(),
        num_examples=len(frames),
        local_head=(updated_head,) if personalize_head else None,
    )


# Familiar spelling used by some callers.
fed_avg = fedavg
fed_prox_step = fedprox_step
fedprox = fedprox_step
fed_prox = fedprox_step
