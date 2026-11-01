"""Versioned raw JSONL storage with atomic completion and validated resume."""

from collections.abc import Sequence
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

from face_recognition.domain.models.dataset import (
    CONDITIONS, COORDINATE_SPACE, SCHEMA_VERSION, CaptureProtocol, RawLandmarkSample,
    SessionConfig, SessionManifest, validate_raw_landmarks,
)
from face_recognition.domain.models.face_landmarks import FaceLandmarks, Landmark


def _reject_constant(value: str) -> None:
    raise ValueError("Non-finite JSON numeric constants are not allowed")


def sample_to_json(sample: RawLandmarkSample) -> str:
    """Keep original numeric values and ordered x/y/z triples; never round."""
    validate_raw_landmarks(sample.landmarks)
    return json.dumps({
        "schema_version": SCHEMA_VERSION,
        "person_id": sample.person_id, "session_id": sample.session_id,
        "condition": sample.condition, "sample_index": sample.sample_index,
        "detector_timestamp_ms": sample.detector_timestamp_ms,
        "inference_width": sample.inference_width, "inference_height": sample.inference_height,
        "landmarks": [[point.x, point.y, point.z] for point in sample.landmarks.points],
    }, allow_nan=False, separators=(",", ":"))


def sample_from_json(line: str) -> RawLandmarkSample:
    try:
        data = json.loads(line, parse_constant=_reject_constant)
        if not isinstance(data, dict) or type(data.get("schema_version")) is not int or data["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Unsupported raw-sample schema version")
        points = data["landmarks"]
        if not isinstance(points, list) or not points or any(not isinstance(point, list) or len(point) != 3 for point in points):
            raise ValueError("Landmarks must be a nonempty list of ordered x/y/z triples")
        return RawLandmarkSample(
            data["person_id"], data["session_id"], data["condition"], data["sample_index"],
            data["detector_timestamp_ms"], data["inference_width"], data["inference_height"],
            FaceLandmarks(tuple(Landmark(*point) for point in points)),
        )
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid raw landmark sample JSON/schema") from exc


def manifest_to_dict(manifest: SessionManifest) -> dict[str, Any]:
    config = asdict(manifest.config)
    protocol = config.pop("protocol")
    return {
        "schema_version": SCHEMA_VERSION, **config, "created_at": manifest.created_at,
        "landmark_coordinate_space": COORDINATE_SPACE,
        "detector_timestamp_clock": "Monotonic submission milliseconds; spacing within each condition/capture process",
        "expected_landmark_count": manifest.expected_landmark_count,
        "conditions": list(CONDITIONS), **protocol,
        "completed_conditions": list(manifest.completed_conditions),
    }


def manifest_from_dict(data: dict[str, Any]) -> SessionManifest:
    try:
        if type(data.get("schema_version")) is not int or data["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Unsupported session schema version")
        if data["conditions"] != list(CONDITIONS) or data["landmark_coordinate_space"] != COORDINATE_SPACE:
            raise ValueError("Session condition protocol or coordinate space is incompatible")
        if not isinstance(data["completed_conditions"], list):
            raise ValueError("Completed conditions must be a JSON list")
        created = datetime.fromisoformat(data["created_at"])
        if created.utcoffset() is None:
            raise ValueError("Capture creation time requires a timezone")
        protocol = CaptureProtocol(data["warmup_results"], data["samples_per_condition"],
                                   data["minimum_sample_interval_ms"])
        config = SessionConfig(**{name: data[name] for name in SessionConfig.__dataclass_fields__ if name != "protocol"},
                               protocol=protocol)
        return SessionManifest(config, data["created_at"], data["expected_landmark_count"],
                               tuple(data["completed_conditions"]))
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Invalid session manifest schema") from exc


class FilesystemDatasetRepository:
    """One JSONL per complete condition; unfinished in-memory batches restart.

    Files are flushed/fsynced before publication. Atomic hard-link publication
    prevents replacing an existing condition, then removes its temporary name.
    Both target files are on the same filesystem. A complete unmarked file (a
    crash between publication and manifest replacement) is validated/adopted on
    resume. Partial .tmp files never count as completed conditions.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()

    def _directory(self, config: SessionConfig) -> Path:
        person = self.root / config.person_id
        directory = person / config.session_id
        if person.is_symlink() or directory.is_symlink() or not directory.resolve().is_relative_to(self.root):
            raise ValueError("Dataset session paths must stay inside the dataset root without symlinks")
        return directory

    def _file(self, directory: Path, name: str) -> Path:
        path = directory / name
        if path.is_symlink() or not path.resolve().is_relative_to(self.root):
            raise ValueError("Dataset files must stay inside the dataset root without symlinks")
        return path

    @staticmethod
    def _sync_directory(directory: Path) -> None:
        if os.name == "posix":
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

    def _atomic_write(self, path: Path, text: str, exclusive: bool = False) -> None:
        temporary = self._file(path.parent, path.name + ".tmp")
        if temporary.exists():
            temporary.unlink()  # Restart only this temporary write; never truncate a linked final file.
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(temporary, path)  # Atomic no-overwrite publication on NTFS/ext4.
            temporary.unlink()
        else:
            os.replace(temporary, path)
        self._sync_directory(path.parent)

    def _write_manifest(self, manifest: SessionManifest) -> None:
        path = self._file(self._directory(manifest.config), "manifest.json")
        self._atomic_write(path, json.dumps(manifest_to_dict(manifest), allow_nan=False, indent=2) + "\n")

    def _read_manifest(self, directory: Path) -> SessionManifest:
        try:
            with self._file(directory, "manifest.json").open(encoding="utf-8") as stream:
                data = json.load(stream, parse_constant=_reject_constant)
            if not isinstance(data, dict):
                raise ValueError("Session manifest must be a JSON object")
            return manifest_from_dict(data)
        except json.JSONDecodeError as exc:
            raise ValueError("Session manifest contains invalid JSON") from exc

    def open_session(self, config: SessionConfig) -> SessionManifest:
        directory = self._directory(config)
        if not directory.exists():
            directory.mkdir(parents=True)
            manifest = SessionManifest(config, datetime.now(timezone.utc).isoformat())
            self._write_manifest(manifest)
            return manifest
        if not self._file(directory, "manifest.json").is_file():
            raise ValueError("Existing session has no manifest; choose a new session ID or repair the session")
        manifest = self._read_manifest(directory)
        if manifest.config != config:
            differences = [name for name in SessionConfig.__dataclass_fields__
                           if getattr(manifest.config, name) != getattr(config, name)]
            raise ValueError("Incompatible existing session configuration: " + ", ".join(differences))
        for condition in manifest.completed_conditions:
            self.read_condition(manifest, condition)
        # Recover complete files published just before an interrupted manifest update.
        for condition in CONDITIONS[len(manifest.completed_conditions):]:
            if self._file(directory, condition + ".jsonl").exists():
                if condition != CONDITIONS[len(manifest.completed_conditions)]:
                    raise ValueError("Unmarked condition files do not follow protocol order")
                samples = self.read_condition(manifest, condition)
                manifest = replace(manifest, expected_landmark_count=len(samples[0].landmarks.points),
                                   completed_conditions=(*manifest.completed_conditions, condition))
                self._write_manifest(manifest)
        return manifest

    @staticmethod
    def _validate_batch(manifest: SessionManifest, condition: str, samples: Sequence[RawLandmarkSample]) -> None:
        config = manifest.config
        if condition not in CONDITIONS:
            raise ValueError("Unknown capture condition")
        if len(samples) != config.protocol.samples_per_condition:
            raise ValueError("A completed condition requires the full configured sample count")
        count = manifest.expected_landmark_count or len(samples[0].landmarks.points)
        previous_timestamp: int | None = None
        for index, sample in enumerate(samples):
            if (sample.person_id != config.person_id or sample.session_id != config.session_id
                    or sample.condition != condition or sample.sample_index != index
                    or sample.inference_width != config.inference_width or sample.inference_height != config.inference_height):
                raise ValueError("Sample metadata/index does not match the session/condition")
            validate_raw_landmarks(sample.landmarks, count)
            if previous_timestamp is not None:
                interval = sample.detector_timestamp_ms - previous_timestamp
                if interval <= 0 or interval < config.protocol.minimum_sample_interval_ms:
                    raise ValueError("Samples must have increasing, sufficiently spaced detector timestamps")
            previous_timestamp = sample.detector_timestamp_ms

    def read_condition(self, manifest: SessionManifest, condition: str) -> tuple[RawLandmarkSample, ...]:
        if condition not in CONDITIONS:
            raise ValueError("Unknown capture condition")
        path = self._file(self._directory(manifest.config), condition + ".jsonl")
        with path.open(encoding="utf-8") as stream:
            samples = tuple(sample_from_json(line) for line in stream)
        self._validate_batch(manifest, condition, samples)
        return samples

    def save_condition(
        self, manifest: SessionManifest, condition: str, samples: Sequence[RawLandmarkSample],
    ) -> SessionManifest:
        directory = self._directory(manifest.config)
        current = self._read_manifest(directory)
        if current != manifest:
            raise ValueError("Session manifest changed; reopen the session before continuing")
        if len(manifest.completed_conditions) == len(CONDITIONS) or condition != CONDITIONS[len(manifest.completed_conditions)]:
            raise ValueError("Condition is already completed or is out of protocol order")
        self._validate_batch(manifest, condition, samples)
        path = self._file(directory, condition + ".jsonl")
        if path.exists():
            raise ValueError("Condition file already exists; resume to validate it, never overwrite it")
        self._atomic_write(path, "".join(sample_to_json(sample) + "\n" for sample in samples), exclusive=True)
        updated = replace(manifest, expected_landmark_count=len(samples[0].landmarks.points),
                          completed_conditions=(*manifest.completed_conditions, condition))
        self._write_manifest(updated)
        return updated
