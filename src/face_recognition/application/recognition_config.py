"""Validate the frozen operational contract and canonical metadata digests."""

from dataclasses import asdict, replace
from hashlib import sha256
import json
from math import isfinite
from pathlib import PurePosixPath, PureWindowsPath
import re

from face_recognition.domain.models.dynamic_population import DynamicConfiguration, DynamicPopulation, DynamicDatasetProvenance
from face_recognition.domain.models.rejection import RejectionError
from face_recognition.domain.models.v3_open_set_config import (EVIDENCE_KINDS, EVIDENCE_SOURCE_TYPES, IMPLEMENTATION_MODULES, FrozenV3OpenSetConfiguration)


def metadata_sha256(value) -> str:
    """Canonical semantic metadata digest, independent of JSON member ordering."""
    if hasattr(value, "__dataclass_fields__"):
        value = asdict(value)
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _digest(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise RejectionError("Invalid SHA-256 provenance")


def _rate(value):
    if type(value) not in (int, float) or not isfinite(value) or not 0 <= value <= 1:
        raise RejectionError("Evidence metrics must be finite rates")
    return float(value)


def validate_frozen_configuration(config):
    expected = FrozenV3OpenSetConfiguration(config.normalization, config.normalization_sha256, config.implementations,
        config.evidence, config.protocol_sha256, config.protocol_catalog_sha256, config.development_evidence)
    if (metadata_sha256(replace(config, configuration_sha256="")) != metadata_sha256(expected)
            or config.configuration_sha256 != metadata_sha256(expected)
            or config.normalization_sha256 != metadata_sha256(config.normalization)):
        raise RejectionError("Frozen configuration fields/digest are incompatible")
    for values in (config.evidence, config.implementations, config.development_evidence,
                   config.normalization.first_eye, config.normalization.second_eye,
                   config.normalization.superior_reference, config.normalization.inferior_reference):
        if not isinstance(values, tuple):
            raise RejectionError("Frozen metadata must use immutable sequences")
    if tuple(r.kind for r in config.evidence) != EVIDENCE_KINDS or tuple(r.module for r in config.implementations) != IMPLEMENTATION_MODULES:
        raise RejectionError("Missing evidence/implementation fingerprints")
    for row, source_type in zip(config.evidence, EVIDENCE_SOURCE_TYPES):
        _digest(row.sha256)
        name = row.filename
        if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+\.json", name)
                or PureWindowsPath(name).name != name or PurePosixPath(name).name != name or type(row.schema_version) is not int
                or row.schema_version != 1 or row.source_type != source_type):
            raise RejectionError("Evidence references must be technical JSON filenames, not paths")
    for row in config.implementations: _digest(row.sha256)
    for value in (config.protocol_sha256, config.protocol_catalog_sha256, config.normalization_sha256): _digest(value)
    if not config.development_evidence or len({(r.source, r.metric) for r in config.development_evidence}) != len(config.development_evidence):
        raise RejectionError("Missing or duplicate aggregate evidence metrics")
    for row in config.development_evidence:
        if row.source != "pose_report" or not re.fullmatch(r"[a-z0-9_.]+", row.metric):
            raise RejectionError("Unsupported evidence metric")
        _rate(row.value)


def validate_dynamic_configuration(config):
    if (type(config) is not DynamicConfiguration or type(config.population) is not DynamicPopulation
            or type(config.dataset) is not DynamicDatasetProvenance):
        raise RejectionError("Invalid dynamic configuration type")
    expected = DynamicConfiguration(config.population, config.dataset, config.normalization, config.implementations)
    if (metadata_sha256(replace(config, configuration_sha256="")) != metadata_sha256(expected)
            or config.configuration_sha256 != metadata_sha256(expected)
            or tuple(row.module for row in config.implementations) != IMPLEMENTATION_MODULES):
        raise RejectionError("Invalid dynamic configuration or algorithm parameters")
    for row in config.implementations:
        _digest(row.sha256)
    for value in asdict(config.dataset).values():
        _digest(value)


def validate_recognition_configuration(config):
    if type(config) is DynamicConfiguration:
        validate_dynamic_configuration(config)
    else:
        validate_frozen_configuration(config)
