"""A targeted semantic label-flip attack for the synthetic fixture.

The attack changes labels for a selected generator and leaves numeric frame
features untouched.  It is a simple corruption primitive for smoke tests; it
does not inject a visual trigger, model replacement, or adaptive adversary.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Sequence

from .data import FrameRecord, VideoRecord


@dataclass(frozen=True, slots=True)
class LabelFlipAttack:
    """Configuration for a generator-targeted semantic label flip."""

    target_generator: str
    source_label: int = 1
    target_label: int = 0

    def __post_init__(self) -> None:
        if not self.target_generator:
            raise ValueError("target_generator is required")
        if self.source_label not in (0, 1) or self.target_label not in (0, 1):
            raise ValueError("labels must be binary")
        if self.source_label == self.target_label:
            raise ValueError("source_label and target_label must differ")

    def apply(self, videos: Iterable[VideoRecord]) -> tuple[VideoRecord, ...]:
        """Return copied videos with only matching semantic labels changed."""

        return targeted_semantic_label_flip(
            videos,
            target_generator=self.target_generator,
            source_label=self.source_label,
            target_label=self.target_label,
        )

    def apply_to_clients(
        self,
        clients: Mapping[str, Sequence[VideoRecord]],
        *,
        malicious_client_ids: Iterable[str],
    ) -> dict[str, tuple[VideoRecord, ...]]:
        """Apply the attack only to selected client collections."""

        malicious = set(malicious_client_ids)
        return {
            client_id: (
                self.apply(videos) if client_id in malicious else tuple(videos)
            )
            for client_id, videos in clients.items()
        }


def targeted_semantic_label_flip(
    videos: Iterable[VideoRecord],
    *,
    target_generator: str,
    source_label: int = 1,
    target_label: int = 0,
) -> tuple[VideoRecord, ...]:
    """Flip target-generator labels without changing features or metadata.

    Videos whose generator or label does not match remain value-identical.  A
    matching video's frame labels and video label are changed together so
    downstream code cannot accidentally train on contradictory record levels.
    """

    if not target_generator:
        raise ValueError("target_generator is required")
    if source_label not in (0, 1) or target_label not in (0, 1):
        raise ValueError("labels must be binary")
    if source_label == target_label:
        raise ValueError("source_label and target_label must differ")
    changed: list[VideoRecord] = []
    for video in videos:
        should_flip = (
            video.generator == target_generator and video.label == source_label
        )
        if not should_flip:
            changed.append(video)
            continue
        frames = tuple(
            replace(frame, label=target_label) for frame in video.frames
        )
        changed.append(replace(video, label=target_label, frames=frames))
    return tuple(changed)


def targeted_label_flip(
    videos: Iterable[VideoRecord],
    target_generator: str,
    *,
    source_label: int = 1,
    target_label: int = 0,
) -> tuple[VideoRecord, ...]:
    """Positional convenience alias for the targeted semantic attack."""

    return targeted_semantic_label_flip(
        videos,
        target_generator=target_generator,
        source_label=source_label,
        target_label=target_label,
    )
