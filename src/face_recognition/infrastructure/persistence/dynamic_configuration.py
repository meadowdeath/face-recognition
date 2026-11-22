"""Canonical dynamic configuration IO and current implementation checks."""

from dataclasses import asdict, replace
import json

from face_recognition.application.recognition_config import metadata_sha256, validate_dynamic_configuration
from face_recognition.domain.models.dataset import CaptureProtocol
from face_recognition.domain.models.dynamic_population import DynamicConfiguration, DynamicDatasetProvenance, DynamicPopulation
from face_recognition.domain.models.rejection import RejectionError
from face_recognition.infrastructure.persistence.artifact_integrity import _object, _read_json
from face_recognition.infrastructure.persistence.recognition_config import current_implementation_fingerprints, current_pose_normalization


def parse_population(raw):
    row = _object(raw, DynamicPopulation)
    for key in ("known_ids", "unknown_ids", "known_sessions", "unknown_sessions"):
        if not isinstance(row[key], list):
            raise RejectionError("Population assignments must be JSON arrays")
        row[key] = tuple(row[key])
    row["capture_protocol"] = CaptureProtocol(**_object(row["capture_protocol"], CaptureProtocol))
    return DynamicPopulation(**row)


def read_population(path):
    return parse_population(_read_json(path))


def dynamic_configuration(population, provenance):
    config = DynamicConfiguration(population, provenance, current_pose_normalization(),
                                  current_implementation_fingerprints())
    return replace(config, configuration_sha256=metadata_sha256(config))


def configuration_bytes(config):
    validate_dynamic_configuration(config)
    return (json.dumps(asdict(config), sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def read_dynamic_configuration(path):
    raw = _object(_read_json(path), DynamicConfiguration)
    config = dynamic_configuration(parse_population(raw["population"]),
                                   DynamicDatasetProvenance(**_object(raw["dataset"], DynamicDatasetProvenance)))
    validate_dynamic_configuration(config)
    if metadata_sha256(raw) != metadata_sha256(config):
        raise RejectionError("Dynamic configuration digest, policy or implementation fingerprints differ")
    return config
