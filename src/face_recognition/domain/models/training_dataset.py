"""Authenticated population/catalog metadata for chronological training."""

from dataclasses import dataclass
import re
from face_recognition.domain.models.dataset import CONDITIONS, validate_identifier
from face_recognition.domain.models.processed_sample import BASELINE_REPRESENTATION
from face_recognition.domain.models.rejection import RejectionError


@dataclass(frozen=True)
class TrainingSpecification:
    known_ids: tuple[str, ...]
    unknown_ids: tuple[str, ...]
    known_sessions: tuple[str, ...]
    unknown_sessions: tuple[str, ...]
    experiment_id: str
    representation: str = BASELINE_REPRESENTATION
    schema_version: int = 1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1 or self.representation != BASELINE_REPRESENTATION:
            raise RejectionError("Unsupported V3 experiment specification")
        validate_identifier(self.experiment_id)
        for name in ("known_ids", "unknown_ids", "known_sessions", "unknown_sessions"):
            values = getattr(self, name)
            if not isinstance(values, (tuple, list)) or not values:
                raise RejectionError("Explicit nonempty identity/session assignments are required")
            for value in values:
                validate_identifier(value)
            if len(set(values)) != len(values):
                raise RejectionError("Duplicate identity/session assignment")
            object.__setattr__(self, name, tuple(sorted(values)))
        if set(self.known_ids) & set(self.unknown_ids):
            raise RejectionError("KNOWN and UNKNOWN identity roles must be disjoint")
        if len(self.known_ids) < 2 or len(self.known_sessions) != 4 or len(self.unknown_sessions) != 2:
            raise RejectionError("Require at least two KNOWN identities, four known sessions and two unknown sessions")
        if len(self.unknown_ids) != 8:
            raise RejectionError("Frozen training population requires exactly eight UNKNOWN identities")


@dataclass(frozen=True)
class ObservedSessions:
    person_id: str
    session_ids: tuple[str, ...]


@dataclass(frozen=True)
class CatalogFile:
    person_id: str
    session_id: str
    condition: str
    source_relative_path: str
    size_bytes: int


@dataclass(frozen=True)
class TrainingCatalog:
    representation: str
    processing_manifest_sha256: str
    declared_source_samples: int
    observed_sessions: tuple[ObservedSessions, ...]
    condition_files: tuple[CatalogFile, ...]


@dataclass(frozen=True)
class TrainingProtocol:
    specification: TrainingSpecification
    catalog: TrainingCatalog
    protocol_sha256: str
    catalog_sha256: str

    def __post_init__(self):
        catalog, spec = self.catalog, self.specification
        if (catalog.representation != BASELINE_REPRESENTATION
                or type(catalog.declared_source_samples) is not int
                or catalog.declared_source_samples < len(catalog.condition_files)
                or any(not isinstance(v, str) or not re.fullmatch(r"[a-f0-9]{64}", v)
                       for v in (self.protocol_sha256, self.catalog_sha256, catalog.processing_manifest_sha256))):
            raise RejectionError("Invalid frozen catalog metadata")
        expected = {person: spec.known_sessions for person in spec.known_ids}
        expected.update({person: spec.unknown_sessions for person in spec.unknown_ids})
        if tuple((r.person_id, r.session_ids) for r in catalog.observed_sessions) != tuple(sorted(expected.items())):
            raise RejectionError("Frozen identity/session assignments differ from catalog")
        keys = tuple((p, s, c) for p, sessions in sorted(expected.items()) for s in sessions for c in CONDITIONS)
        if tuple((r.person_id, r.session_id, r.condition) for r in catalog.condition_files) != keys:
            raise RejectionError("Frozen catalog requires every ordered condition")
        for r in catalog.condition_files:
            if (r.source_relative_path != f"{BASELINE_REPRESENTATION}/{r.person_id}/{r.session_id}/{r.condition}.jsonl"
                    or type(r.size_bytes) is not int or r.size_bytes < 1):
                raise RejectionError("Invalid frozen catalog file reference")
