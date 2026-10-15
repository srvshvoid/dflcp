"""Compatibility-facing names for the synthetic data boundary.

The implementation lives in :mod:`fedforge.data`; this module keeps imports
readable for experiment code that calls the fixture ``synthetic`` data.
"""

from .data import (
    FrameRecord,
    LeakageError,
    LeakageReport,
    SyntheticConfig,
    VideoRecord,
    assert_client_partition_no_leakage,
    assert_no_leakage,
    check_no_leakage,
    check_partition_leakage,
    generate_synthetic_videos,
    make_synthetic_videos,
    partition_by_client,
    partition_clients,
    partition_videos,
    split_train_test,
    split_videos,
)

__all__ = [
    "FrameRecord",
    "LeakageError",
    "LeakageReport",
    "SyntheticConfig",
    "VideoRecord",
    "assert_client_partition_no_leakage",
    "assert_no_leakage",
    "check_no_leakage",
    "check_partition_leakage",
    "generate_synthetic_videos",
    "make_synthetic_videos",
    "partition_by_client",
    "partition_clients",
    "partition_videos",
    "split_train_test",
    "split_videos",
]
