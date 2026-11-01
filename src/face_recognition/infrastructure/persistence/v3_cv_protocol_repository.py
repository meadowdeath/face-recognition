"""Metadata-only V3 catalog inspection and atomic immutable protocol persistence."""

from dataclasses import asdict, fields
import json
from pathlib import Path

from face_recognition.application.create_v3_cv_protocol import metadata_sha256, validate_v3_cv_protocol
from face_recognition.domain.models.processed_sample import BASELINE_REPRESENTATION
from face_recognition.domain.models.v3_cv_protocol import (
    V3CVProtocol, V3Catalog, V3CatalogFile, V3ExperimentSpecification, V3InnerSplit,
    V3ObservedSessions, V3OuterFold, V3ProtocolError, V3ProtocolGeneration,
)
from face_recognition.infrastructure.persistence.aggregate_report import write_aggregate_report_json
from face_recognition.infrastructure.persistence.artifact_integrity import file_sha256
from face_recognition.infrastructure.persistence.processed_dataset_catalog import ProcessedDatasetCatalog, reject_json_constant


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise V3ProtocolError("Duplicate metadata JSON key")
        result[key] = value
    return result


def _read_json(path):
    if Path(path).is_symlink():
        raise V3ProtocolError("Metadata source must not be a symlink")
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_object,
                      parse_constant=reject_json_constant)


def _object(value, model):
    if not isinstance(value, dict) or set(value) != {f.name for f in fields(model)}:
        raise V3ProtocolError("Incomplete or unexpected V3 metadata fields")
    return dict(value)


def _array(value):
    if not isinstance(value, list):
        raise V3ProtocolError("V3 metadata sequence must be a JSON array")
    return tuple(value)


def _specification(value):
    data = _object(value, V3ExperimentSpecification)
    for name in ("known_ids", "unknown_ids", "known_sessions", "unknown_sessions"):
        data[name] = _array(data[name])
    return V3ExperimentSpecification(**data)


def read_v3_experiment_specification(path: Path) -> V3ExperimentSpecification:
    try:
        return _specification(_read_json(path))
    except Exception:
        raise V3ProtocolError("Invalid V3 experiment specification or explicit assignments") from None


def read_v3_catalog(processed_root: Path) -> V3Catalog:
    """Inspect all_xyz_v1 directories, file sizes and completed processing metadata.

    Never opens/hashes JSONL payloads. Catalog validity is not record validity:
    source row counts, feature width/values and indices must be checked by a
    future payload loader before model execution. Other representations are not
    inputs to this protocol and are not inspected.
    """
    try:
        if Path(processed_root).is_symlink():
            raise ValueError
        source = ProcessedDatasetCatalog(processed_root, BASELINE_REPRESENTATION)
        manifest_path = source.root / "processing_manifest.json"
        before = file_sha256(manifest_path)
        metadata = _read_json(manifest_path)
        if metadata != source._manifest:
            raise ValueError
        keys = source.discover_conditions()
        observed = []
        # Explicit directory inspection also rejects empty/unassigned directories
        # that would not contribute any keys to the shared condition catalog.
        for person in source._entries(source.root):
            if person.name == "processing_manifest.json":
                continue
            observed.append(V3ObservedSessions(person.name, tuple(s.name for s in source._entries(person))))
        records = []
        for person, session, condition in keys:
            path = source._checked(source.root / person / session / (condition + ".jsonl"))
            records.append(V3CatalogFile(person, session, condition,
                           path.relative_to(source.processed_root).as_posix(), path.stat().st_size))
        if (file_sha256(manifest_path) != before or source.discover_conditions() != keys
                or any((source.processed_root / row.source_relative_path).stat().st_size != row.size_bytes for row in records)):
            raise ValueError
        return V3Catalog(BASELINE_REPRESENTATION, before, metadata["total_samples_processed"], tuple(observed), tuple(records))
    except Exception:
        raise V3ProtocolError("Invalid/incomplete or changing all_xyz_v1 processed catalog") from None


def v3_protocol_to_json(protocol: V3CVProtocol) -> str:
    validate_v3_cv_protocol(protocol)
    return json.dumps(asdict(protocol), sort_keys=True, allow_nan=False, indent=2) + "\n"


def write_v3_cv_protocol(protocol: V3CVProtocol, output: Path, inputs: tuple[Path, ...]) -> None:
    write_aggregate_report_json(v3_protocol_to_json(protocol), output, inputs)


def read_v3_cv_protocol(path: Path) -> V3CVProtocol:
    try:
        raw = _read_json(path)
        data = _object(raw, V3CVProtocol)
        data["specification"] = _specification(data["specification"])
        catalog = _object(data["catalog"], V3Catalog)
        observed = []
        for item in _array(catalog["observed_sessions"]):
            row = _object(item, V3ObservedSessions)
            row["session_ids"] = _array(row["session_ids"])
            observed.append(V3ObservedSessions(**row))
        catalog["observed_sessions"] = tuple(observed)
        catalog["condition_files"] = tuple(V3CatalogFile(**_object(item, V3CatalogFile)) for item in _array(catalog["condition_files"]))
        data["catalog"] = V3Catalog(**catalog)
        folds = []
        for item in _array(data["outer_folds"]):
            outer = _object(item, V3OuterFold)
            for name in ("known_development_sessions", "unknown_development_ids", "unknown_test_ids"):
                outer[name] = _array(outer[name])
            splits = []
            for item in _array(outer["inner_splits"]):
                inner = _object(item, V3InnerSplit)
                for name in ("known_fit_sessions", "unknown_calibration_ids", "unknown_validation_ids"):
                    inner[name] = _array(inner[name])
                splits.append(V3InnerSplit(**inner))
            outer["inner_splits"] = tuple(splits)
            folds.append(V3OuterFold(**outer))
        data["outer_folds"] = tuple(folds)
        data["generation"] = V3ProtocolGeneration(**_object(data["generation"], V3ProtocolGeneration))
        protocol = V3CVProtocol(**data)
        validate_v3_cv_protocol(protocol)
        if metadata_sha256(raw) != metadata_sha256(protocol):
            raise ValueError
        return protocol
    except Exception:
        raise V3ProtocolError("Invalid, changed or incomplete V3 protocol metadata/digests") from None
