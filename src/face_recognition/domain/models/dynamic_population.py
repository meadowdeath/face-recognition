"""Versioned population and algorithm contract for dynamic construction."""

from dataclasses import dataclass, field

from face_recognition.domain.models.dataset import CONDITIONS, CaptureProtocol, validate_identifier
from face_recognition.domain.models.processed_sample import POSE_REPRESENTATION
from face_recognition.domain.models.training_dataset import TrainingCatalog
from face_recognition.domain.models.rejection import RejectionError
from face_recognition.domain.models.v3_open_set_config import (
    FrozenPoseNormalization, FrozenV3Pipeline, FrozenV3TemporalPolicy, FrozenV3ThresholdPolicy,
    ImplementationFingerprint,
)


MODEL_ID = "open_set_model_dynamic"
KNOWN_SESSIONS = ("session_01", "session_02", "session_03", "session_04")
UNKNOWN_SESSIONS = KNOWN_SESSIONS[:2]
POLICY = "minimum_one_unknown_per_two_known"
LIMITATIONS = (
    "Population ratio is a project policy, not a guarantee of FAR or accuracy.",
    "Calibration metrics are not independent test or generalization results.",
)


@dataclass(frozen=True)
class DynamicPopulation:
    known_ids: tuple[str, ...]
    unknown_ids: tuple[str, ...]
    known_sessions: tuple[str, ...] = KNOWN_SESSIONS
    unknown_sessions: tuple[str, ...] = UNKNOWN_SESSIONS
    capture_protocol: CaptureProtocol = field(default_factory=CaptureProtocol)
    representation: str = POSE_REPRESENTATION
    population_policy: str = POLICY
    schema_version: int = 1

    def __post_init__(self):
        if (type(self.schema_version) is not int or self.schema_version != 1
                or self.population_policy != POLICY or self.representation != POSE_REPRESENTATION
                or self.known_sessions != KNOWN_SESSIONS or self.unknown_sessions != UNKNOWN_SESSIONS
                or type(self.capture_protocol) is not CaptureProtocol or self.capture_protocol != CaptureProtocol()):
            raise RejectionError("Unsupported dynamic population schema, representation, sessions or capture protocol")
        assigned = []
        for field_name in ("known_ids", "unknown_ids"):
            values = getattr(self, field_name)
            if not isinstance(values, (list, tuple)):
                raise RejectionError("Identity assignments must be arrays")
            for identity in values:
                validate_identifier(identity)
                assigned.append(identity.lower())
            object.__setattr__(self, field_name, tuple(sorted(values)))
        if len(set(assigned)) != len(assigned):
            raise RejectionError("Duplicate or overlapping identities, including case-insensitive path collisions")
        known, unknown = len(self.known_ids), len(self.unknown_ids)
        if known < 2 or unknown < self.minimum_unknown:
            raise RejectionError(f"Invalid population: {known} KNOWN (minimum 2), {unknown} UNKNOWN "
                                 f"(minimum {self.minimum_unknown} for this KNOWN count)")

    @property
    def minimum_unknown(self):
        return (len(self.known_ids) + 1) // 2

    @property
    def assignments(self):
        return {**dict.fromkeys(self.known_ids, self.known_sessions),
                **dict.fromkeys(self.unknown_ids, self.unknown_sessions)}

    @property
    def condition_keys(self):
        return tuple((person, session, condition) for person, sessions in sorted(self.assignments.items())
                     for session in sessions for condition in CONDITIONS)


@dataclass(frozen=True)
class DynamicTrainingProtocol:
    specification: DynamicPopulation
    catalog: TrainingCatalog
    catalog_sha256: str


@dataclass(frozen=True)
class DynamicDatasetProvenance:
    raw_dataset_sha256: str
    processed_files_sha256: str
    catalog_sha256: str
    run_manifest_sha256: str
    representation_manifest_sha256: str
    payload_sha256: str


@dataclass(frozen=True)
class DynamicConfiguration:
    population: DynamicPopulation
    dataset: DynamicDatasetProvenance
    normalization: FrozenPoseNormalization
    implementations: tuple[ImplementationFingerprint, ...]
    configuration_sha256: str = ""
    schema_version: int = 1
    configuration_id: str = "dynamic_population_config"
    pipeline: FrozenV3Pipeline = field(default_factory=FrozenV3Pipeline)
    temporal: FrozenV3TemporalPolicy = field(default_factory=FrozenV3TemporalPolicy)
    threshold_policy: FrozenV3ThresholdPolicy = field(default_factory=FrozenV3ThresholdPolicy)
    limitations: tuple[str, ...] = LIMITATIONS


@dataclass(frozen=True)
class DynamicTrainingProvenance:
    configuration_sha256: str
    dataset_sha256: str
