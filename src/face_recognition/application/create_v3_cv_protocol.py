"""Pure deterministic partition planning. No data loading, fitting or selection."""

from dataclasses import asdict, replace
from hashlib import sha256
from itertools import permutations
import json
import re

from face_recognition.domain.models.dataset import CONDITIONS
from face_recognition.domain.models.processed_sample import BASELINE_REPRESENTATION
from face_recognition.domain.models.v3_cv_protocol import (
    V3CVProtocol, V3Catalog, V3ExperimentSpecification, V3InnerSplit, V3OuterFold, V3ProtocolError,
)


def metadata_sha256(value) -> str:
    """Canonical semantic metadata digest, independent of JSON member ordering."""
    if hasattr(value, "__dataclass_fields__"):
        value = asdict(value)
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def validate_catalog(spec: V3ExperimentSpecification, catalog: V3Catalog) -> None:
    if (catalog.representation != BASELINE_REPRESENTATION
            or not re.fullmatch(r"[a-f0-9]{64}", catalog.processing_manifest_sha256)
            or type(catalog.declared_source_samples) is not int
            or catalog.declared_source_samples < len(catalog.condition_files)):
        raise V3ProtocolError("Invalid completed representation catalog metadata")
    expected = {person: spec.known_sessions for person in spec.known_ids}
    expected.update({person: spec.unknown_sessions for person in spec.unknown_ids})
    observed = tuple((row.person_id, row.session_ids) for row in catalog.observed_sessions)
    if observed != tuple(sorted(expected.items())):
        raise V3ProtocolError("Unassigned/missing identities or missing/extra configured sessions")
    expected_keys = tuple((person, session, condition) for person, sessions in sorted(expected.items())
                          for session in sessions for condition in CONDITIONS)
    actual_keys = tuple((row.person_id, row.session_id, row.condition) for row in catalog.condition_files)
    if actual_keys != expected_keys:
        raise V3ProtocolError("Each configured session requires the complete ordered condition catalog")
    for row in catalog.condition_files:
        relative = f"{BASELINE_REPRESENTATION}/{row.person_id}/{row.session_id}/{row.condition}.jsonl"
        if row.source_relative_path != relative or type(row.size_bytes) is not int or row.size_bytes < 1:
            raise V3ProtocolError("Invalid relative source reference or empty condition file")


def create_v3_cv_protocol(spec: V3ExperimentSpecification, catalog: V3Catalog) -> V3CVProtocol:
    """Four outer tests and 18 three-role inner splits per outer fold.

    All KNOWN identities participate in every known session role. Each UNKNOWN
    role selects all of that identity's cataloged sessions/conditions. Calibration
    and validation are separate, including on the KNOWN side. Inner validation
    compares candidates, never recalibrates their thresholds. Any later outer
    refit/calibration must remain inside that outer fold's development roles.
    """
    # Revalidate even for callers constructing dataclasses without a repository.
    spec = V3ExperimentSpecification(**asdict(spec))
    validate_catalog(spec, catalog)
    groups = tuple(spec.unknown_ids[i:i + 2] for i in range(0, len(spec.unknown_ids), 2))
    outer = []
    for index, test_session in enumerate(spec.known_sessions):
        dev_sessions = tuple(s for s in spec.known_sessions if s != test_session)
        dev_ids = tuple(person for person in spec.unknown_ids if person not in groups[index])
        validation_groups = tuple(group for i, group in enumerate(groups) if i != index)
        inner = []
        for fit, calibration, validation in permutations(dev_sessions):
            for validation_ids in validation_groups:
                inner.append(V3InnerSplit(
                    f"inner_{len(inner) + 1:02d}", (fit,), calibration, validation,
                    tuple(person for person in dev_ids if person not in validation_ids), validation_ids,
                ))
        outer.append(V3OuterFold(f"outer_{index + 1:02d}", dev_sessions, test_session,
                                 dev_ids, groups[index], tuple(inner)))
    protocol = V3CVProtocol(spec, catalog, tuple(outer), metadata_sha256(spec), metadata_sha256(catalog))
    payload = asdict(protocol)
    del payload["protocol_sha256"]
    return replace(protocol, protocol_sha256=metadata_sha256(payload))


def validate_v3_cv_protocol(protocol: V3CVProtocol) -> None:
    """Regenerate all roles as well as hashes; rehashed illegal folds still fail."""
    expected = create_v3_cv_protocol(protocol.specification, protocol.catalog)
    if metadata_sha256(protocol) != metadata_sha256(expected):
        raise V3ProtocolError("Protocol metadata/digest differs from the deterministic V3 generation policy")
