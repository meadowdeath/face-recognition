"""Atomic, no-overwrite publication of external metadata-only split manifests."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile

from face_recognition.config.settings import PROJECT_ROOT
from face_recognition.application.validate_identity_split import validate_identity_split
from face_recognition.domain.models.identity_split import (
    IdentityPartition, IdentitySplitManifest, IdentitySplitSpec, PartitionCounts, SessionPartition, SplitSampleReference,
)
from face_recognition.infrastructure.persistence.processed_dataset_catalog import reject_json_constant


class IdentitySplitPersistenceError(ValueError):
    pass


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate split JSON keys")
        result[key] = value
    return result


def _fields(data, model):
    if not isinstance(data, dict) or set(data) != set(model.__dataclass_fields__):
        raise ValueError("Invalid split schema fields")
    return data


def read_identity_split(path: Path) -> IdentitySplitManifest:
    """Load and strictly validate metadata only; never consult feature files."""
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            data = json.load(stream, parse_constant=reject_json_constant, object_pairs_hook=_unique_object)
        _fields(data, IdentitySplitManifest)
        spec_data = _fields(data["specification"], IdentitySplitSpec)
        if any(not isinstance(value, list) for value in spec_data.values()):
            raise ValueError
        spec = IdentitySplitSpec(**{name: tuple(value) for name, value in spec_data.items()})
        if any(tuple(spec_data[name]) != getattr(spec, name) for name in spec_data):
            raise ValueError("Noncanonical split specification")
        assignments = tuple(SessionPartition(**{**_fields(row, SessionPartition),
                                                 "partition": IdentityPartition(row["partition"])})
                            for row in data["session_assignments"])
        counts = tuple(PartitionCounts(**{**_fields(row, PartitionCounts),
                                          "partition": IdentityPartition(row["partition"])})
                       for row in data["partition_counts"])
        samples = tuple(SplitSampleReference(**{**_fields(row, SplitSampleReference),
                                                "partition": IdentityPartition(row["partition"])})
                        for row in data["samples"])
        values = {**data, "specification": spec, "session_assignments": assignments,
                  "partition_counts": counts, "samples": samples,
                  "partition_definitions": tuple(data["partition_definitions"]),
                  "leakage_constraints": tuple(data["leakage_constraints"])}
        manifest = IdentitySplitManifest(**values)
        validate_identity_split(manifest)
        return manifest
    except Exception:
        raise IdentitySplitPersistenceError(f"Invalid or unreadable identity split manifest: {Path(path)}") from None


def write_identity_split(manifest: IdentitySplitManifest, output: Path, processed_root: Path) -> None:
    if Path(output).is_symlink():
        raise IdentitySplitPersistenceError("Split output must not be a symlink")
    target, source = Path(output).resolve(), Path(processed_root).resolve()
    if target.is_relative_to(PROJECT_ROOT.resolve()):
        raise IdentitySplitPersistenceError("Split artifacts must remain outside the Git repository")
    if target.is_relative_to(source) or source.is_relative_to(target):
        raise IdentitySplitPersistenceError("Split output must be outside the processed source tree")
    if target.exists():
        raise IdentitySplitPersistenceError("Split output already exists; choose a new output path")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(asdict(manifest), allow_nan=False, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
