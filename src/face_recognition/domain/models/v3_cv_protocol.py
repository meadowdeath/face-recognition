"""Metadata-only nested session/identity protocol; no models or biometric values."""

from dataclasses import dataclass, field

from face_recognition.domain.models.dataset import validate_identifier
from face_recognition.domain.models.processed_sample import BASELINE_REPRESENTATION


class V3ProtocolError(ValueError):
    """Invalid experimental roles, catalog or protocol integrity."""


@dataclass(frozen=True)
class V3ExperimentSpecification:
    known_ids: tuple[str, ...]
    unknown_ids: tuple[str, ...]
    known_sessions: tuple[str, ...]
    unknown_sessions: tuple[str, ...]
    experiment_id: str
    representation: str = BASELINE_REPRESENTATION
    schema_version: int = 1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1 or self.representation != BASELINE_REPRESENTATION:
            raise V3ProtocolError("Unsupported V3 experiment specification")
        validate_identifier(self.experiment_id)
        for name in ("known_ids", "unknown_ids", "known_sessions", "unknown_sessions"):
            values = getattr(self, name)
            if not isinstance(values, (tuple, list)) or not values:
                raise V3ProtocolError("Explicit nonempty identity/session assignments are required")
            for value in values:
                validate_identifier(value)
            if len(set(values)) != len(values):
                raise V3ProtocolError("Duplicate identity/session assignment")
            object.__setattr__(self, name, tuple(sorted(values)))
        if set(self.known_ids) & set(self.unknown_ids):
            raise V3ProtocolError("KNOWN and UNKNOWN identity roles must be disjoint")
        if len(self.known_ids) < 2 or len(self.known_sessions) != 4 or len(self.unknown_sessions) != 2:
            raise V3ProtocolError("Require at least two KNOWN identities, four known sessions and two unknown sessions")
        if len(self.unknown_ids) != 8:
            raise V3ProtocolError("Four outer groups of two UNKNOWN identities require exactly eight identities")


@dataclass(frozen=True)
class V3ObservedSessions:
    person_id: str
    session_ids: tuple[str, ...]


@dataclass(frozen=True)
class V3CatalogFile:
    person_id: str
    session_id: str
    condition: str
    source_relative_path: str
    size_bytes: int


@dataclass(frozen=True)
class V3Catalog:
    representation: str
    processing_manifest_sha256: str
    declared_source_samples: int
    observed_sessions: tuple[V3ObservedSessions, ...]
    condition_files: tuple[V3CatalogFile, ...]


@dataclass(frozen=True)
class V3InnerSplit:
    name: str
    known_fit_sessions: tuple[str, ...]
    known_calibration_session: str
    known_validation_session: str
    unknown_calibration_ids: tuple[str, ...]
    unknown_validation_ids: tuple[str, ...]


@dataclass(frozen=True)
class V3OuterFold:
    name: str
    known_development_sessions: tuple[str, ...]
    known_test_session: str
    unknown_development_ids: tuple[str, ...]
    unknown_test_ids: tuple[str, ...]
    inner_splits: tuple[V3InnerSplit, ...]


@dataclass(frozen=True)
class V3ProtocolGeneration:
    version: int = 1
    outer_fold_count: int = 4
    unknown_outer_group_size: int = 2
    inner_policy: str = "All six fit/calibration/validation session permutations crossed with three UNKNOWN-pair rotations"
    unknown_grouping: str = "Lexical IDs in consecutive disjoint pairs; inner validation rotates the three remaining outer groups"
    fitting_scope: str = "KNOWN inner-fit sessions only; UNKNOWN never fits preprocessing, identity classifier or novelty references"
    threshold_scope: str = "KNOWN inner-calibration session plus UNKNOWN inner-calibration identities only"
    selection_scope: str = "Inner validation only; no per-inner-validation threshold adjustment"
    outer_scope: str = "Outer test is inaccessible to fitting/calibration/selection; later outer refit and threshold construction use outer development only"
    repeated_observations: str = "Inner splits reuse development observations; do not count repeats as independent physical samples"


@dataclass(frozen=True)
class V3CVProtocol:
    specification: V3ExperimentSpecification
    catalog: V3Catalog
    outer_folds: tuple[V3OuterFold, ...]
    specification_sha256: str
    catalog_sha256: str
    protocol_sha256: str = ""
    generation: V3ProtocolGeneration = field(default_factory=V3ProtocolGeneration)
    schema_version: int = 1
    protocol: str = "v3_nested_session_identity_cv"
    completion_state: str = "catalog_validated"
    ordering: str = "Lexical identities/sessions; capture-protocol conditions; outer test session order then inner session permutations/pair order"
    payload_validation: str = "No feature file opened; catalog completeness only. Validate records/dimensions/provenance before any future fitting or scoring."
