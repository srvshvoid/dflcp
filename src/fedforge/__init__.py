"""FedForge's dependency-free CPU research slice.

The package contains a deliberately small, deterministic implementation of
the contracts needed by the smoke experiment.  It is useful for exercising
data-splitting, aggregation, attack, metric, and persistence boundaries; it
is not a claim of a production federated-learning or media-forensics stack.
"""

from .attack import LabelFlipAttack, targeted_semantic_label_flip
from .data import (
    FrameRecord,
    LeakageError,
    LeakageReport,
    SyntheticConfig,
    VideoRecord,
    assert_no_leakage,
    check_no_leakage,
    generate_synthetic_videos,
    partition_clients,
    split_train_test,
)
from .federation import (
    ClientUpdate,
    FederatedState,
    fedavg,
    fedprox_step,
    personalized_aggregate,
)
from .metrics import (
    accuracy,
    auc,
    bootstrap_over_videos,
    ece,
    f1,
    per_generator_recall,
)
from .model import (
    LogisticFrameModel,
    aggregate_video_predictions,
    mean_video_probability,
)
from .payload import PayloadAuditError, audit_payload
from .robust import (
    CompatibilityError,
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
    "FrameRecord",
    "LabelFlipAttack",
    "LeakageError",
    "LeakageReport",
    "LogisticFrameModel",
    "PayloadAuditError",
    "SyntheticConfig",
    "VideoRecord",
    "accuracy",
    "aggregate_video_predictions",
    "assert_no_leakage",
    "audit_payload",
    "auc",
    "bootstrap_over_videos",
    "check_no_leakage",
    "coordinate_median",
    "ece",
    "f1",
    "fedavg",
    "fedprox_step",
    "generate_synthetic_videos",
    "krum",
    "mean_video_probability",
    "norm_clip",
    "partition_clients",
    "per_generator_recall",
    "personalized_aggregate",
    "split_train_test",
    "targeted_semantic_label_flip",
    "trimmed_mean",
    "validate_secagg_compatibility",
]
