"""A tiny typed logistic frame model used by the CPU smoke simulator.

This is intentionally a linear classifier over already extracted numeric
features.  It is an interface-compatible baseline for the vertical slice,
not a CNN/ViT forensic model and not evidence about real media performance.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

from .data import FrameRecord, VideoRecord


def _sigmoid(value: float) -> float:
    """Compute a numerically stable logistic sigmoid."""

    if value >= 0.0:
        exponential = math.exp(-value)
        return 1.0 / (1.0 + exponential)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("feature and parameter dimensions do not match")
    return sum(a * b for a, b in zip(left, right))


@dataclass(frozen=True, slots=True)
class LogisticFrameModel:
    """A binary logistic classifier with immutable parameter tuples.

    ``parameters()`` returns weights followed by bias, which is the shared
    tensor sent in a federated update.  A personalized local head is modeled
    separately as an additive scalar by the federation module and is never
    part of this shared parameter vector.
    """

    weights: tuple[float, ...]
    bias: float = 0.0

    def __post_init__(self) -> None:
        normalized = tuple(float(value) for value in self.weights)
        if not normalized:
            raise ValueError("a logistic model needs at least one weight")
        object.__setattr__(self, "weights", normalized)
        object.__setattr__(self, "bias", float(self.bias))

    @classmethod
    def zeros(cls, n_features: int) -> "LogisticFrameModel":
        """Create an all-zero model for deterministic initialization."""

        if n_features < 1:
            raise ValueError("n_features must be positive")
        return cls(weights=tuple(0.0 for _ in range(n_features)), bias=0.0)

    @classmethod
    def from_parameters(cls, parameters: Sequence[float]) -> "LogisticFrameModel":
        """Build a model from ``weights + bias`` parameters."""

        if len(parameters) < 2:
            raise ValueError("at least one weight and one bias are required")
        return cls(weights=tuple(float(value) for value in parameters[:-1]), bias=float(parameters[-1]))

    def parameters(self) -> tuple[float, ...]:
        """Return the shared parameter vector in a stable order."""

        return self.weights + (self.bias,)

    @property
    def n_features(self) -> int:
        """Return the feature dimension."""

        return len(self.weights)

    def predict_logit(
        self,
        features: Sequence[float],
        *,
        local_head: float = 0.0,
    ) -> float:
        """Return a frame logit, optionally including a private head bias."""

        return _dot(self.weights, features) + self.bias + local_head

    def predict_proba(
        self,
        features: Sequence[float],
        *,
        local_head: float = 0.0,
    ) -> float:
        """Return the positive-class frame probability."""

        return _sigmoid(self.predict_logit(features, local_head=local_head))

    def predict(self, features: Sequence[float], *, threshold: float = 0.5, local_head: float = 0.0) -> int:
        """Return a binary frame prediction."""

        return int(self.predict_proba(features, local_head=local_head) >= threshold)

    def predict_frame_proba(
        self,
        features: Sequence[float],
        *,
        local_head: float = 0.0,
    ) -> float:
        """Readable alias for :meth:`predict_proba` at frame level."""

        return self.predict_proba(features, local_head=local_head)

    def predict_video_proba(
        self,
        video: VideoRecord,
        *,
        local_head: float = 0.0,
    ) -> float:
        """Aggregate frame probabilities by their arithmetic mean."""

        return mean_video_probability(self, video, local_head=local_head)

    def predict_video(
        self,
        video: VideoRecord,
        *,
        threshold: float = 0.5,
        local_head: float = 0.0,
    ) -> int:
        """Return a thresholded video prediction after mean aggregation."""

        return int(self.predict_video_proba(video, local_head=local_head) >= threshold)

    def loss_and_gradient(
        self,
        frames: Sequence[FrameRecord],
        *,
        local_head: float = 0.0,
        proximal_reference: Sequence[float] | None = None,
        proximal_mu: float = 0.0,
    ) -> tuple[float, tuple[float, ...], float]:
        """Return mean logistic loss, shared gradient, and local-head gradient.

        The optional proximal term is ``mu / 2 * ||theta - theta_global||²``.
        It affects only shared parameters; the local head remains client-local.
        """

        if not frames:
            raise ValueError("cannot train on an empty frame collection")
        if proximal_mu < 0.0:
            raise ValueError("proximal_mu must be non-negative")
        parameters = self.parameters()
        if proximal_reference is not None and len(proximal_reference) != len(parameters):
            raise ValueError("proximal reference dimension does not match model")
        gradient = [0.0 for _ in parameters]
        head_gradient = 0.0
        loss = 0.0
        for frame in frames:
            probability = self.predict_proba(frame.features, local_head=local_head)
            error = probability - frame.label
            for index, feature in enumerate(frame.features):
                gradient[index] += error * feature
            gradient[-1] += error
            head_gradient += error
            # Clamp only the log operation; the probability itself remains the
            # exact sigmoid result used by the gradient.
            loss -= frame.label * math.log(max(probability, 1e-15))
            loss -= (1 - frame.label) * math.log(max(1.0 - probability, 1e-15))
        scale = 1.0 / len(frames)
        gradient = [value * scale for value in gradient]
        head_gradient *= scale
        loss *= scale
        if proximal_reference is not None and proximal_mu:
            for index, (parameter, reference) in enumerate(zip(parameters, proximal_reference)):
                delta = parameter - reference
                gradient[index] += proximal_mu * delta
                loss += 0.5 * proximal_mu * delta * delta
        return loss, tuple(gradient), head_gradient

    def fit_frames(
        self,
        frames: Iterable[FrameRecord],
        *,
        epochs: int = 1,
        learning_rate: float = 0.1,
        local_head: float = 0.0,
        personalize_head: bool = False,
        proximal_reference: Sequence[float] | None = None,
        proximal_mu: float = 0.0,
    ) -> tuple["LogisticFrameModel", float, float]:
        """Perform deterministic full-batch gradient descent on frame records.

        Returns ``(updated_shared_model, updated_local_head, final_loss)``.
        The local head is updated only when ``personalize_head`` is true.  A
        caller can therefore retain one head per client while aggregating only
        shared parameters.
        """

        frame_list = tuple(frames)
        if not frame_list:
            raise ValueError("cannot train on an empty frame collection")
        if epochs < 1 or learning_rate <= 0.0:
            raise ValueError("epochs must be positive and learning_rate must be positive")
        if proximal_mu < 0.0:
            raise ValueError("proximal_mu must be non-negative")
        model = self
        head = float(local_head)
        final_loss = 0.0
        for _ in range(epochs):
            final_loss, gradient, head_gradient = model.loss_and_gradient(
                frame_list,
                local_head=head,
                proximal_reference=proximal_reference,
                proximal_mu=proximal_mu,
            )
            parameters = tuple(
                parameter - learning_rate * value
                for parameter, value in zip(model.parameters(), gradient)
            )
            model = LogisticFrameModel.from_parameters(parameters)
            if personalize_head:
                head -= learning_rate * head_gradient
        return model, head, final_loss


def mean_video_probability(
    model: LogisticFrameModel,
    video: VideoRecord,
    *,
    local_head: float = 0.0,
) -> float:
    """Return the mean positive probability across a video's frames."""

    probabilities = [
        model.predict_proba(frame.features, local_head=local_head)
        for frame in video.frames
    ]
    return sum(probabilities) / len(probabilities)


def aggregate_video_predictions(frame_probabilities: Sequence[float]) -> float:
    """Mean-pool precomputed frame probabilities into one video score."""

    if not frame_probabilities:
        raise ValueError("at least one frame probability is required")
    if any(not 0.0 <= probability <= 1.0 for probability in frame_probabilities):
        raise ValueError("frame probabilities must be in [0, 1]")
    return sum(frame_probabilities) / len(frame_probabilities)


aggregate_video_probability = aggregate_video_predictions
video_mean = aggregate_video_predictions


def train_logistic_model(
    frames: Iterable[FrameRecord],
    *,
    n_features: int | None = None,
    epochs: int = 5,
    learning_rate: float = 0.1,
) -> LogisticFrameModel:
    """Convenience baseline trainer for a collection of synthetic frames."""

    frame_list = tuple(frames)
    if not frame_list and n_features is None:
        raise ValueError("n_features is required when frames are empty")
    dimension = n_features if n_features is not None else len(frame_list[0].features)
    model = LogisticFrameModel.zeros(dimension)
    model, _, _ = model.fit_frames(
        frame_list,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    return model
