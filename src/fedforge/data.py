"""Deterministic synthetic records and leakage-safe video partitions.

The records in this module contain numeric features only.  They intentionally
stand in for extracted frame features, rather than pretending to be images or
video files.  A source is the unit used to make train/test and client splits
disjoint: keeping frames from one source together prevents an overly
optimistic source-leakage result in the smoke experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class FrameRecord:
    """One deterministic feature record for one video frame.

    ``features`` are synthetic numeric descriptors.  No raw bytes, image
    objects, or filenames are represented by this core record.
    """

    video_id: str
    source_id: str
    generator: str
    frame_index: int
    features: tuple[float, ...]
    label: int

    def __post_init__(self) -> None:
        if not self.video_id or not self.source_id or not self.generator:
            raise ValueError("video_id, source_id, and generator are required")
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if not self.features:
            raise ValueError("features must not be empty")
        if self.label not in (0, 1):
            raise ValueError("the smoke classifier uses binary labels 0 and 1")


@dataclass(frozen=True, slots=True)
class VideoRecord:
    """A video-level grouping of frame feature records.

    The label and generator are repeated at video and frame level so callers
    can evaluate video-level metrics without carrying media metadata around.
    """

    video_id: str
    source_id: str
    generator: str
    label: int
    frames: tuple[FrameRecord, ...]

    def __post_init__(self) -> None:
        if not self.video_id or not self.source_id or not self.generator:
            raise ValueError("video_id, source_id, and generator are required")
        if self.label not in (0, 1):
            raise ValueError("the smoke classifier uses binary labels 0 and 1")
        if not self.frames:
            raise ValueError("a video must contain at least one frame")
        for frame in self.frames:
            if (
                frame.video_id != self.video_id
                or frame.source_id != self.source_id
                or frame.generator != self.generator
            ):
                raise ValueError("frame metadata must match its containing video")

    @property
    def frame_count(self) -> int:
        """Return the number of synthetic frames in the video."""

        return len(self.frames)


@dataclass(frozen=True, slots=True)
class SyntheticConfig:
    """Configuration for :func:`generate_synthetic_videos`.

    This fixture is for deterministic CPU smoke coverage.  Its distributions
    are not a dataset or a benchmark claim.
    """

    seed: int = 7
    n_videos: int = 24
    frames_per_video: int = 4
    n_features: int = 4
    source_count: int | None = 8
    generators: tuple[str, ...] = ("real", "gan", "diffusion")

    def __post_init__(self) -> None:
        if self.n_videos < 2:
            raise ValueError("n_videos must be at least two")
        if self.frames_per_video < 1 or self.n_features < 1:
            raise ValueError("frames_per_video and n_features must be positive")
        if self.source_count is not None and self.source_count < 2:
            raise ValueError("source_count must be at least two when specified")
        if not self.generators or len(set(self.generators)) != len(self.generators):
            raise ValueError("generators must contain unique names")


def _config_with_overrides(
    config: SyntheticConfig | None,
    *,
    seed: int | None,
    n_videos: int | None,
    frames_per_video: int | None,
    n_features: int | None,
    source_count: int | None,
    generators: Sequence[str] | None,
    count: int | None,
) -> SyntheticConfig:
    base = config or SyntheticConfig()
    return SyntheticConfig(
        seed=base.seed if seed is None else seed,
        n_videos=base.n_videos if count is None and n_videos is None else (count or n_videos or base.n_videos),
        frames_per_video=base.frames_per_video if frames_per_video is None else frames_per_video,
        n_features=base.n_features if n_features is None else n_features,
        source_count=base.source_count if source_count is None else source_count,
        generators=base.generators if generators is None else tuple(generators),
    )


def generate_synthetic_videos(
    config: SyntheticConfig | None = None,
    *,
    seed: int | None = None,
    n_videos: int | None = None,
    frames_per_video: int | None = None,
    n_features: int | None = None,
    source_count: int | None = None,
    generators: Sequence[str] | None = None,
    count: int | None = None,
) -> tuple[VideoRecord, ...]:
    """Generate reproducible video/frame records.

    The random generator is local to the call, so generation does not mutate
    process-global random state.  ``count`` is accepted as a concise alias for
    ``n_videos`` for small scripts.
    """

    cfg = _config_with_overrides(
        config,
        seed=seed,
        n_videos=n_videos,
        frames_per_video=frames_per_video,
        n_features=n_features,
        source_count=source_count,
        generators=generators,
        count=count,
    )
    rng = random.Random(cfg.seed)
    source_count_value = cfg.source_count or cfg.n_videos
    source_count_value = min(source_count_value, cfg.n_videos)
    source_offsets: dict[str, tuple[float, ...]] = {}
    for source_index in range(source_count_value):
        source_id = f"source-{source_index:03d}"
        source_offsets[source_id] = tuple(
            rng.uniform(-0.25, 0.25) for _ in range(cfg.n_features)
        )

    records: list[VideoRecord] = []
    for video_index in range(cfg.n_videos):
        source_id = f"source-{video_index % source_count_value:03d}"
        generator = cfg.generators[video_index % len(cfg.generators)]
        label = 0 if generator == cfg.generators[0] else 1
        video_id = f"video-{video_index:04d}"
        generator_offset = (cfg.generators.index(generator) - 0.5) * 0.18
        frames: list[FrameRecord] = []
        for frame_index in range(cfg.frames_per_video):
            values: list[float] = []
            for feature_index in range(cfg.n_features):
                class_signal = (1.15 if label else -1.15) if feature_index == 0 else 0.0
                generator_signal = generator_offset if feature_index == 1 else 0.0
                source_signal = source_offsets[source_id][feature_index]
                noise = rng.uniform(-0.22, 0.22)
                values.append(class_signal + generator_signal + source_signal + noise)
            frames.append(
                FrameRecord(
                    video_id=video_id,
                    source_id=source_id,
                    generator=generator,
                    frame_index=frame_index,
                    features=tuple(values),
                    label=label,
                )
            )
        records.append(
            VideoRecord(
                video_id=video_id,
                source_id=source_id,
                generator=generator,
                label=label,
                frames=tuple(frames),
            )
        )
    return tuple(records)


@dataclass(frozen=True, slots=True)
class LeakageReport:
    """Identifiers shared by two supposedly disjoint partitions."""

    video_overlap: tuple[str, ...]
    source_overlap: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """Whether both video and source identifiers are disjoint."""

        return not self.video_overlap and not self.source_overlap


class LeakageError(ValueError):
    """Raised when a split or client partition leaks an identifier."""


def check_no_leakage(
    left: Iterable[VideoRecord],
    right: Iterable[VideoRecord],
    *,
    require_source_disjoint: bool = True,
) -> LeakageReport:
    """Return overlap details for two video collections.

    A shared source is reported even when video IDs differ.  Set
    ``require_source_disjoint=False`` only for a deliberate diagnostic where
    source reuse is acceptable; video IDs are always checked.
    """

    left_videos = tuple(left)
    right_videos = tuple(right)
    left_video_ids = {video.video_id for video in left_videos}
    right_video_ids = {video.video_id for video in right_videos}
    left_sources = {video.source_id for video in left_videos}
    right_sources = {video.source_id for video in right_videos}
    return LeakageReport(
        video_overlap=tuple(sorted(left_video_ids & right_video_ids)),
        source_overlap=(
            tuple(sorted(left_sources & right_sources))
            if require_source_disjoint
            else tuple()
        ),
    )


def assert_no_leakage(
    left: Iterable[VideoRecord],
    right: Iterable[VideoRecord],
    *,
    require_source_disjoint: bool = True,
) -> LeakageReport:
    """Validate two partitions and raise :class:`LeakageError` on overlap."""

    report = check_no_leakage(
        left, right, require_source_disjoint=require_source_disjoint
    )
    if report.video_overlap or (
        require_source_disjoint and report.source_overlap
    ):
        raise LeakageError(
            "partition leakage: "
            f"videos={report.video_overlap!r}, sources={report.source_overlap!r}"
        )
    return report


def _source_groups(videos: Iterable[VideoRecord]) -> dict[str, list[VideoRecord]]:
    groups: dict[str, list[VideoRecord]] = {}
    seen_video_ids: set[str] = set()
    for video in videos:
        if video.video_id in seen_video_ids:
            raise LeakageError(f"duplicate video ID: {video.video_id}")
        seen_video_ids.add(video.video_id)
        groups.setdefault(video.source_id, []).append(video)
    return groups


def split_train_test(
    videos: Sequence[VideoRecord],
    *,
    test_fraction: float = 0.25,
    seed: int = 0,
    source_disjoint: bool = True,
) -> tuple[tuple[VideoRecord, ...], tuple[VideoRecord, ...]]:
    """Split videos deterministically, keeping sources together by default."""

    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between zero and one")
    if len(videos) < 2:
        raise ValueError("at least two videos are required")
    if not source_disjoint:
        shuffled = list(videos)
        random.Random(seed).shuffle(shuffled)
        test_count = max(1, min(len(shuffled) - 1, int(len(shuffled) * test_fraction)))
        test_ids = {video.video_id for video in shuffled[:test_count]}
    else:
        groups = _source_groups(videos)
        if len(groups) < 2:
            raise ValueError("source-disjoint splitting requires at least two sources")
        source_ids = list(groups)
        random.Random(seed).shuffle(source_ids)
        test_group_count = max(1, min(len(source_ids) - 1, int(len(source_ids) * test_fraction)))
        test_sources = set(source_ids[:test_group_count])
        test_ids = {
            video.video_id for source_id, group in groups.items() if source_id in test_sources for video in group
        }
    train = tuple(video for video in videos if video.video_id not in test_ids)
    test = tuple(video for video in videos if video.video_id in test_ids)
    assert_no_leakage(train, test, require_source_disjoint=source_disjoint)
    return train, test


def partition_clients(
    videos: Sequence[VideoRecord],
    n_clients: int,
    *,
    seed: int = 0,
    source_disjoint: bool = True,
) -> dict[str, tuple[VideoRecord, ...]]:
    """Partition videos across named clients without splitting a video.

    With ``source_disjoint=True`` (the default), all videos from one source
    are assigned to one client.  A greedy balancing pass keeps the synthetic
    fixture reasonably even while preserving that invariant.
    """

    if n_clients < 1:
        raise ValueError("n_clients must be positive")
    if not videos:
        raise ValueError("at least one video is required")
    clients: dict[str, list[VideoRecord]] = {
        f"client-{index}": [] for index in range(n_clients)
    }
    rng = random.Random(seed)
    if source_disjoint:
        groups = list(_source_groups(videos).values())
        rng.shuffle(groups)
        groups.sort(key=len, reverse=True)
        for group in groups:
            target = min(clients, key=lambda client_id: (len(clients[client_id]), client_id))
            clients[target].extend(group)
    else:
        shuffled = list(videos)
        rng.shuffle(shuffled)
        for index, video in enumerate(shuffled):
            clients[f"client-{index % n_clients}"].append(video)
    result = {client_id: tuple(items) for client_id, items in clients.items()}
    all_items = [video for items in result.values() for video in items]
    if len({video.video_id for video in all_items}) != len(all_items):
        raise LeakageError("a video was assigned to more than one client")
    if source_disjoint:
        client_items = list(result.items())
        for index, (_, left) in enumerate(client_items):
            for _, right in client_items[index + 1 :]:
                assert_no_leakage(left, right, require_source_disjoint=True)
    return result


def assert_client_partition_no_leakage(
    clients: Mapping[str, Sequence[VideoRecord]],
    *,
    require_source_disjoint: bool = True,
) -> None:
    """Raise if any two client collections share video/source identifiers."""

    items = list(clients.items())
    for index, (_, left) in enumerate(items):
        for _, right in items[index + 1 :]:
            assert_no_leakage(
                left, right, require_source_disjoint=require_source_disjoint
            )


# Short aliases are useful in notebooks and preserve the same strict defaults.
partition_videos = partition_clients
partition_by_client = partition_clients
split_videos = split_train_test
make_synthetic_videos = generate_synthetic_videos
check_partition_leakage = assert_client_partition_no_leakage
