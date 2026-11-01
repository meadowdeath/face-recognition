"""Plan/validate unknown identity roles using source metadata only."""

from dataclasses import asdict
from hashlib import sha256
import json

from face_recognition.application.validate_identity_split import validate_identity_split
from face_recognition.domain.models.identity_split import IdentityPartition, IdentitySplitManifest
from face_recognition.domain.models.unknown_rejection import UnknownPartitionCounts, UnknownRejectionSplit, UnknownSplitSpec


class UnknownRejectionError(ValueError):
    pass


def plan_unknown_split(source: IdentitySplitManifest, spec: UnknownSplitSpec) -> UnknownRejectionSplit:
    validate_identity_split(source)
    ids = set(spec.calibration_ids) | set(spec.holdout_ids)
    if ids != set(source.specification.unknown_ids):
        raise UnknownRejectionError("Assign every unknown_pool identity exactly once; known or absent IDs are not allowed")

    def counts(selected: tuple[str, ...]) -> UnknownPartitionCounts:
        rows = [row for row in source.samples if row.partition is IdentityPartition.UNKNOWN_POOL and row.person_id in selected]
        return UnknownPartitionCounts(len({r.person_id for r in rows}), len({(r.person_id, r.session_id) for r in rows}),
                                      len({r.source_relative_path for r in rows}), len(rows))

    digest = sha256(json.dumps(asdict(source), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    return UnknownRejectionSplit(spec, digest, source.schema_version, source.total_sample_count,
                                 counts(spec.calibration_ids), counts(spec.holdout_ids))


def validate_unknown_split(source: IdentitySplitManifest, split: UnknownRejectionSplit) -> None:
    expected = plan_unknown_split(source, split.specification)
    # Canonical JSON comparison also distinguishes booleans from integer counts.
    if json.dumps(asdict(split), sort_keys=True) != json.dumps(asdict(expected), sort_keys=True):
        raise UnknownRejectionError("Unknown rejection split does not match its source identity split or counts")
