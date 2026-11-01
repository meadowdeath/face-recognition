"""Logical sample references and session-level leakage boundaries; no feature values."""

from dataclasses import dataclass
from enum import Enum

from face_recognition.domain.models.dataset import validate_identifier
from face_recognition.domain.models.processed_sample import BASELINE_REPRESENTATION, SampleProvenance


class IdentityPartition(str, Enum):
    KNOWN_TRAIN_POOL = "known_train_pool"
    KNOWN_TEST = "known_test"
    UNKNOWN_POOL = "unknown_pool"


@dataclass(frozen=True)
class FeatureSampleMetadata:
    provenance: SampleProvenance
    landmark_count: int
    feature_count: int

    def __post_init__(self) -> None:
        if (type(self.landmark_count) is not int or self.landmark_count < 1
                or type(self.feature_count) is not int or self.feature_count != 3 * self.landmark_count):
            raise ValueError("Invalid baseline feature dimensions")


@dataclass(frozen=True)
class IdentitySplitSpec:
    known_ids: tuple[str, ...]
    unknown_ids: tuple[str, ...]
    train_sessions: tuple[str, ...]
    known_test_sessions: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("known_ids", "unknown_ids", "train_sessions", "known_test_sessions"):
            values = tuple(getattr(self, name))
            for value in values:
                validate_identifier(value)
            if len(values) != len(set(values)):
                raise ValueError(f"Duplicate entries in {name}")
            object.__setattr__(self, name, tuple(sorted(values)))
        if not self.known_ids and not self.unknown_ids:
            raise ValueError("Assign at least one identity role explicitly")
        if set(self.known_ids) & set(self.unknown_ids):
            raise ValueError("An identity cannot be both known and unknown")
        if not self.train_sessions or not self.known_test_sessions:
            raise ValueError("Specify nonempty training-pool and known-test session sets")
        if set(self.train_sessions) & set(self.known_test_sessions):
            raise ValueError("Known training-pool and known-test sessions must be disjoint")


@dataclass(frozen=True)
class SessionPartition:
    person_id: str
    session_id: str
    partition: IdentityPartition


@dataclass(frozen=True)
class SplitSampleReference:
    person_id: str
    session_id: str
    condition: str
    sample_index: int
    partition: IdentityPartition
    source_relative_path: str


@dataclass(frozen=True)
class PartitionCounts:
    partition: IdentityPartition
    identities: int
    sessions: int
    condition_files: int
    samples: int


@dataclass(frozen=True)
class IdentitySplitManifest:
    specification: IdentitySplitSpec
    known_identity_count: int
    unknown_identity_count: int
    session_assignments: tuple[SessionPartition, ...]
    partition_counts: tuple[PartitionCounts, ...]
    total_sample_count: int
    samples: tuple[SplitSampleReference, ...]
    schema_version: int = 1
    source_representation: str = BASELINE_REPRESENTATION
    completion_state: str = "completed"
    source_path_base: str = "processed_root"
    ordering: str = "person_id and session_id lexical order, capture condition order, then sample_index"
    partition_definitions: tuple[str, ...] = (
        "known_train_pool: configured complete training sessions of known identities",
        "known_test: configured complete held-out sessions of known identities; final known evaluation only",
        "unknown_pool: all sessions of explicitly unknown identities; reserved, not a final test set",
    )
    leakage_constraints: tuple[str, ...] = (
        "Fit feature selection only on known_train_pool",
        "Derive correlation filtering only from known_train_pool",
        "Fit PCA only on known_train_pool",
        "Fit StandardScaler only on known_train_pool",
        "Fit classifiers only on known_train_pool",
        "Keep known_test out of fitting, feature selection, preprocessing and model/hyperparameter selection",
        "Keep unknown_pool untouched until the later unknown-rejection stage; it is not a final test set",
        "Perform future session-aware validation/model selection inside known_train_pool only",
    )
