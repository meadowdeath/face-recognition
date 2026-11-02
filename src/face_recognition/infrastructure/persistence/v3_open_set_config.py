"""Freeze from aggregate JSON only; atomically publish immutable non-biometric metadata."""

from dataclasses import asdict
from math import isfinite
from pathlib import Path
import json

from face_recognition.application.create_v3_cv_protocol import metadata_sha256
from face_recognition.application.freeze_v3_open_set_config import freeze_v3_open_set_configuration, validate_frozen_v3_configuration
from face_recognition.domain.models.rejection import RejectionError
from face_recognition.domain.models.v3_open_set_config import (
    EVIDENCE_KINDS, IMPLEMENTATION_MODULES, DevelopmentMetricEvidence, FrozenEvidenceReference, FrozenPoseNormalization,
    FrozenV3OpenSetConfiguration, FrozenV3Pipeline, FrozenV3TemporalPolicy, FrozenV3ThresholdPolicy, ImplementationFingerprint,
)
from face_recognition.infrastructure.features import pose_landmark_normalizer
from face_recognition.infrastructure.persistence.aggregate_report import validate_aggregate_report_target, write_aggregate_report_json
from face_recognition.infrastructure.persistence.artifact_integrity import file_sha256
from face_recognition.infrastructure.persistence.v3_cv_protocol_repository import _object, _read_json, read_v3_cv_protocol
from face_recognition.infrastructure.persistence.v3_mahalanobis_calibration_report import read_mahalanobis_selection_report
from face_recognition.infrastructure.persistence.v3_mahalanobis_diagnostics_report import read_mahalanobis_evaluation_report


def current_pose_normalization():
    row = dict(pose_landmark_normalizer.POSE_NORMALIZATION)
    for field in ("first_eye", "second_eye", "superior_reference", "inferior_reference"):
        row[field] = tuple(row[field])
    return FrozenPoseNormalization(**row)


def current_implementation_fingerprints():
    pose = Path(pose_landmark_normalizer.__file__)
    return tuple(ImplementationFingerprint(module, file_sha256(path)) for module, path in
                 zip(IMPLEMENTATION_MODULES, (pose, pose.with_name("mahalanobis.py"))))


def _finite_metadata(value):
    if isinstance(value, dict):
        for child in value.values(): _finite_metadata(child)
    elif isinstance(value, list):
        for child in value: _finite_metadata(child)
    elif isinstance(value, float) and not isfinite(value):
        raise RejectionError("Non-finite evidence metadata")


def load_v3_freeze_evidence(paths):
    """No processed-root argument, JSONL decoder, fitting or inference capability."""
    try:
        if tuple(paths) != EVIDENCE_KINDS:
            raise ValueError
        files = tuple(Path(paths[k]) for k in EVIDENCE_KINDS)
        if (len({p.resolve() for p in files}) != len(files)
                or any(p.is_symlink() or not p.is_file() or p.suffix != ".json" for p in files)):
            raise ValueError
        before = tuple(file_sha256(p) for p in files)
        protocol = read_v3_cv_protocol(files[0])
        # Reuse completed-study validators. These read aggregate metadata only.
        selection = read_mahalanobis_selection_report(files[1], protocol)
        calibration, digest = read_mahalanobis_evaluation_report(files[2], protocol, before[0])
        if selection.report_sha256 != before[1] or digest != before[2] or calibration.selection_provenance != selection:
            raise ValueError
        documents = {kind: _read_json(paths[kind]) for kind in EVIDENCE_KINDS[1:]}
        for row in documents.values(): _finite_metadata(row)
        references = (FrozenEvidenceReference("protocol", files[0].name, before[0], protocol.schema_version, protocol.protocol),
            *(FrozenEvidenceReference(kind, path.name, sha, documents[kind]["schema_version"], documents[kind]["experiment"])
              for kind, path, sha in zip(EVIDENCE_KINDS[1:], files[1:], before[1:])))
        config = freeze_v3_open_set_configuration(protocol, documents, references, current_pose_normalization(), current_implementation_fingerprints())
        if tuple(file_sha256(p) for p in files) != before:
            raise ValueError
        return config
    except Exception:
        raise RejectionError("Cannot freeze V3: malformed/incomplete evidence, incompatible configuration or provenance mismatch") from None


def frozen_v3_configuration_to_json(config):
    validate_frozen_v3_configuration(config)
    if (config.normalization_sha256 != metadata_sha256(current_pose_normalization())
            or config.normalization != current_pose_normalization()
            or config.implementations != current_implementation_fingerprints()):
        raise RejectionError("Frozen normalization/Mahalanobis implementation differs from current V3 source")
    return json.dumps(asdict(config), sort_keys=True, allow_nan=False, indent=2) + "\n"


def write_frozen_v3_configuration(config, output, inputs):
    write_aggregate_report_json(frozen_v3_configuration_to_json(config), output, tuple(inputs))


def freeze_v3_configuration_file(paths, output):
    inputs = tuple(Path(paths[k]) for k in EVIDENCE_KINDS)
    target = validate_aggregate_report_target(output, inputs)  # Refuse before reading evidence.
    config = load_v3_freeze_evidence(paths)
    if any(file_sha256(path) != reference.sha256 for path, reference in zip(inputs, config.evidence)):
        raise RejectionError("Evidence changed before frozen configuration publication")
    write_frozen_v3_configuration(config, target, inputs)
    return config


def read_frozen_v3_configuration(path):
    """Validate configuration/provenance integrity without reading datasets or estimators."""
    try:
        before = file_sha256(Path(path))
        row = _object(_read_json(path), FrozenV3OpenSetConfiguration)
        normalization = _object(row["normalization"], FrozenPoseNormalization)
        for name in ("first_eye", "second_eye", "superior_reference", "inferior_reference"):
            normalization[name] = tuple(normalization[name])
        row["normalization"] = FrozenPoseNormalization(**normalization)
        pipeline = _object(row["pipeline"], FrozenV3Pipeline)
        pipeline["svm_parameters"] = tuple(tuple(pair) for pair in pipeline["svm_parameters"])
        row["pipeline"] = FrozenV3Pipeline(**pipeline)
        temporal = _object(row["temporal"], FrozenV3TemporalPolicy)
        temporal["grouping"] = tuple(temporal["grouping"])
        row["temporal"] = FrozenV3TemporalPolicy(**temporal)
        threshold = _object(row["threshold_policy"], FrozenV3ThresholdPolicy)
        for name in ("fitted_components", "tie_breakers"):
            threshold[name] = tuple(threshold[name])
        row["threshold_policy"] = FrozenV3ThresholdPolicy(**threshold)
        for name, model in (("evidence", FrozenEvidenceReference), ("implementations", ImplementationFingerprint), ("development_evidence", DevelopmentMetricEvidence)):
            row[name] = tuple(model(**_object(value, model)) for value in row[name])
        row["limitations"] = tuple(row["limitations"])
        config = FrozenV3OpenSetConfiguration(**row)
        frozen_v3_configuration_to_json(config)
        if file_sha256(Path(path)) != before:
            raise ValueError
        return config
    except Exception:
        raise RejectionError("Invalid/incompatible frozen V3 configuration or integrity digest") from None
