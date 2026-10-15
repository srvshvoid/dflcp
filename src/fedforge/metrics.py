"""Dependency-free binary and video-level evaluation metrics.

All functions consume already aggregated numeric predictions.  They are
small, transparent implementations suitable for smoke checks; they do not
replace a validated scientific metrics package for large-scale reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable, Generic, Iterable, Sequence, TypeVar

from .data import VideoRecord


def _binary_predictions(values: Sequence[int | float], threshold: float = 0.5) -> list[int]:
    return [int(value >= threshold) for value in values]


def _check_lengths(*values: Sequence[object]) -> None:
    if not values:
        return
    length = len(values[0])
    if length == 0:
        raise ValueError("metric inputs must not be empty")
    if any(len(value) != length for value in values[1:]):
        raise ValueError("metric inputs must have equal lengths")


def accuracy(
    labels: Sequence[int],
    predictions: Sequence[int | float],
    *,
    threshold: float = 0.5,
) -> float:
    """Return binary accuracy, thresholding probability-like predictions."""

    _check_lengths(labels, predictions)
    predicted = _binary_predictions(predictions, threshold)
    return sum(int(label == guess) for label, guess in zip(labels, predicted)) / len(labels)


def f1(
    labels: Sequence[int],
    predictions: Sequence[int | float],
    *,
    threshold: float = 0.5,
) -> float:
    """Return the positive-class binary F1 score."""

    _check_lengths(labels, predictions)
    predicted = _binary_predictions(predictions, threshold)
    true_positive = sum(int(label == 1 and guess == 1) for label, guess in zip(labels, predicted))
    false_positive = sum(int(label == 0 and guess == 1) for label, guess in zip(labels, predicted))
    false_negative = sum(int(label == 1 and guess == 0) for label, guess in zip(labels, predicted))
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else 2.0 * true_positive / denominator


def auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Return ROC AUC via positive/negative pair concordance.

    When only one class is present, ``0.5`` is returned as an explicit
    no-discrimination sentinel instead of inventing a class-specific result.
    """

    _check_lengths(labels, scores)
    positives = [float(score) for label, score in zip(labels, scores) if label == 1]
    negatives = [float(score) for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        return 0.5
    concordant = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                concordant += 1.0
            elif positive == negative:
                concordant += 0.5
    return concordant / (len(positives) * len(negatives))


def ece(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    n_bins: int = 10,
) -> float:
    """Return expected calibration error for equally spaced confidence bins."""

    _check_lengths(labels, probabilities)
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    buckets: list[list[tuple[int, float]]] = [[] for _ in range(n_bins)]
    for label, probability in zip(labels, probabilities):
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probabilities must be in [0, 1]")
        bucket = min(n_bins - 1, int(probability * n_bins))
        buckets[bucket].append((int(label), probability))
    total = len(labels)
    return sum(
        len(bucket) / total
        * abs(
            sum(label for label, _ in bucket) / len(bucket)
            - sum(probability for _, probability in bucket) / len(bucket)
        )
        for bucket in buckets
        if bucket
    )


def per_generator_recall(
    labels_or_videos: Sequence[int] | Sequence[VideoRecord],
    predictions: Sequence[int | float],
    generators: Sequence[str] | None = None,
    *,
    threshold: float = 0.5,
    positive_label: int = 1,
) -> dict[str, float]:
    """Return positive recall separately for each generator.

    Callers may provide ``(labels, predictions, generators)`` or
    ``(videos, predictions)``.  A generator with no positive examples has a
    recall of ``0.0`` and is still present in the output.
    """

    _check_lengths(labels_or_videos, predictions)
    if labels_or_videos and isinstance(labels_or_videos[0], VideoRecord):
        videos = labels_or_videos  # type: ignore[assignment]
        labels = [video.label for video in videos]  # type: ignore[union-attr]
        generator_names = [video.generator for video in videos]  # type: ignore[union-attr]
    else:
        labels = [int(value) for value in labels_or_videos]  # type: ignore[arg-type]
        if generators is None:
            raise ValueError("generators are required when labels are supplied")
        _check_lengths(labels, generators)
        generator_names = list(generators)
    predicted = _binary_predictions(predictions, threshold)
    totals: dict[str, int] = {}
    correct: dict[str, int] = {}
    for label, guess, generator in zip(labels, predicted, generator_names):
        if label == positive_label:
            totals[generator] = totals.get(generator, 0) + 1
            correct[generator] = correct.get(generator, 0) + int(guess == positive_label)
        else:
            totals.setdefault(generator, 0)
            correct.setdefault(generator, 0)
    return {
        generator: (correct[generator] / totals[generator] if totals[generator] else 0.0)
        for generator in sorted(totals)
    }


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """Point estimate and percentile interval from video-level resampling."""

    estimate: float
    lower: float
    upper: float
    samples: tuple[float, ...]
    seed: int
    confidence: float


T = TypeVar("T")


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate a percentile of no values")
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    interpolation = position - lower
    return ordered[lower] + interpolation * (ordered[upper] - ordered[lower])


def bootstrap_over_videos(
    videos: Sequence[T],
    metric: Callable[[Sequence[T]], float],
    *,
    n_resamples: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> BootstrapResult:
    """Bootstrap a metric by resampling whole videos with replacement.

    The callback receives a sequence of sampled video objects, so frames from
    one video cannot be split across bootstrap units.  The local RNG keeps
    this helper deterministic and isolated from process-global state.
    """

    if not videos:
        raise ValueError("videos must not be empty")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    rng = random.Random(seed)
    estimate = float(metric(tuple(videos)))
    samples = tuple(
        float(metric(tuple(videos[rng.randrange(len(videos))] for _ in videos)))
        for _ in range(n_resamples)
    )
    alpha = (1.0 - confidence) / 2.0
    return BootstrapResult(
        estimate=estimate,
        lower=_percentile(samples, alpha),
        upper=_percentile(samples, 1.0 - alpha),
        samples=samples,
        seed=seed,
        confidence=confidence,
    )


# Common spellings retained as explicit aliases for small scripts.
accuracy_score = accuracy
f1_score = f1
roc_auc = auc
expected_calibration_error = ece
generator_recall = per_generator_recall
bootstrap_metric_over_videos = bootstrap_over_videos
