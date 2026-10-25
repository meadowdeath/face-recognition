"""Validate stored split metadata without accessing any feature payloads."""

from face_recognition.domain.models.dataset import CONDITIONS, validate_identifier
from face_recognition.domain.models.identity_split import (
    IdentityPartition, IdentitySplitManifest, IdentitySplitSpec, PartitionCounts, SessionPartition,
)
from face_recognition.domain.models.processed_sample import BASELINE_REPRESENTATION


class SplitValidationError(ValueError):
    pass


def validate_identity_split(manifest: IdentitySplitManifest) -> None:
    if (type(manifest.schema_version) is not int or manifest.schema_version != 1
            or manifest.source_representation != BASELINE_REPRESENTATION
            or manifest.completion_state != "completed" or manifest.source_path_base != "processed_root"):
        raise SplitValidationError("Unsupported or incomplete identity split")
    for name in ("ordering", "partition_definitions", "leakage_constraints"):
        if getattr(manifest, name) != IdentitySplitManifest.__dataclass_fields__[name].default:
            raise SplitValidationError("Split ordering or leakage policy differs from schema version 1")
    spec = manifest.specification
    checked = IdentitySplitSpec(spec.known_ids, spec.unknown_ids, spec.train_sessions, spec.known_test_sessions)
    if checked != spec:
        raise SplitValidationError("Split specification must be canonical")
    for value in (manifest.known_identity_count, manifest.unknown_identity_count, manifest.total_sample_count):
        if type(value) is not int or value < 0:
            raise SplitValidationError("Invalid split aggregate counts")
    if (manifest.known_identity_count != len(spec.known_ids) or manifest.unknown_identity_count != len(spec.unknown_ids)
            or not manifest.samples or manifest.total_sample_count != len(manifest.samples)):
        raise SplitValidationError("Split identity/sample counts are inconsistent")

    known, unknown = set(spec.known_ids), set(spec.unknown_ids)
    sessions = {}
    next_index = {}
    previous = None
    identities = set()
    for sample in manifest.samples:
        validate_identifier(sample.person_id)
        validate_identifier(sample.session_id)
        if (sample.condition not in CONDITIONS or type(sample.sample_index) is not int or sample.sample_index < 0
                or not isinstance(sample.partition, IdentityPartition)):
            raise SplitValidationError("Invalid split sample metadata")
        key = (sample.person_id, sample.session_id, CONDITIONS.index(sample.condition), sample.sample_index)
        condition_key = key[:3]
        if (previous is not None and key <= previous) or sample.sample_index != next_index.get(condition_key, 0):
            raise SplitValidationError("Split sample keys must be unique, ordered and contiguous per condition")
        previous = key
        next_index[condition_key] = sample.sample_index + 1
        expected_path = f"{BASELINE_REPRESENTATION}/{sample.person_id}/{sample.session_id}/{sample.condition}.jsonl"
        if sample.source_relative_path != expected_path:
            raise SplitValidationError("Split source path does not match sample metadata")
        if sample.person_id in unknown:
            expected = IdentityPartition.UNKNOWN_POOL
        elif sample.person_id in known and sample.session_id in spec.train_sessions:
            expected = IdentityPartition.KNOWN_TRAIN_POOL
        elif sample.person_id in known and sample.session_id in spec.known_test_sessions:
            expected = IdentityPartition.KNOWN_TEST
        else:
            raise SplitValidationError("Split sample has an unassigned identity/session")
        if sample.partition != expected:
            raise SplitValidationError("Split partition violates explicit identity/session roles")
        sessions.setdefault((sample.person_id, sample.session_id), expected)
        if sessions[sample.person_id, sample.session_id] != expected:
            raise SplitValidationError("Session occurs in more than one partition")
        identities.add(sample.person_id)
    if identities != known | unknown:
        raise SplitValidationError("Every configured identity must be represented in the split")
    expected_sessions = set(spec.train_sessions) | set(spec.known_test_sessions)
    for person in known:
        if {session for identity, session in sessions if identity == person} != expected_sessions:
            raise SplitValidationError("A known identity lacks configured complete sessions")
    assignments = tuple(SessionPartition(person, session, sessions[person, session]) for person, session in sorted(sessions))
    if manifest.session_assignments != assignments:
        raise SplitValidationError("Session assignments differ from split sample references")
    counts = []
    for row in manifest.partition_counts:
        if any(type(value) is not int or value < 0 for value in (row.identities, row.sessions, row.condition_files, row.samples)):
            raise SplitValidationError("Invalid partition aggregate counts")
    for partition in IdentityPartition:
        selected = [sample for sample in manifest.samples if sample.partition == partition]
        counts.append(PartitionCounts(partition, len({s.person_id for s in selected}),
                                      len({(s.person_id, s.session_id) for s in selected}),
                                      len({s.source_relative_path for s in selected}), len(selected)))
    if manifest.partition_counts != tuple(counts):
        raise SplitValidationError("Partition counts differ from split references")
