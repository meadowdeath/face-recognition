"""Read frozen configuration/provenance without fitting or replaying research."""

from dataclasses import asdict
import json
from pathlib import Path

from face_recognition.application.recognition_config import metadata_sha256, validate_frozen_configuration
from face_recognition.domain.models.rejection import RejectionError
from face_recognition.domain.models.v3_open_set_config import (DevelopmentMetricEvidence, FrozenEvidenceReference, FrozenPoseNormalization, FrozenV3OpenSetConfiguration, FrozenV3Pipeline, FrozenV3TemporalPolicy, FrozenV3ThresholdPolicy, ImplementationFingerprint, IMPLEMENTATION_MODULES)
from face_recognition.infrastructure.features import pose_landmark_normalizer
from face_recognition.infrastructure.persistence.artifact_integrity import file_sha256, _object, _read_json


def current_pose_normalization():
    row = dict(pose_landmark_normalizer.POSE_NORMALIZATION)
    for field in ("first_eye", "second_eye", "superior_reference", "inferior_reference"):
        row[field] = tuple(row[field])
    return FrozenPoseNormalization(**row)


def current_implementation_fingerprints():
    pose = Path(pose_landmark_normalizer.__file__)
    return tuple(ImplementationFingerprint(module, file_sha256(path)) for module, path in
                 zip(IMPLEMENTATION_MODULES, (pose, pose.with_name("mahalanobis.py"))))


def frozen_configuration_to_json(config):
    validate_frozen_configuration(config)
    if (config.normalization_sha256 != metadata_sha256(current_pose_normalization())
            or config.normalization != current_pose_normalization()
            or config.implementations != current_implementation_fingerprints()):
        raise RejectionError("Frozen normalization/Mahalanobis implementation differs from current V3 source")
    return json.dumps(asdict(config), sort_keys=True, allow_nan=False, indent=2) + "\n"


def read_frozen_configuration(path):
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
        frozen_configuration_to_json(config)
        if file_sha256(Path(path)) != before:
            raise ValueError
        return config
    except Exception:
        raise RejectionError("Invalid/incompatible frozen V3 configuration or integrity digest") from None
