"""Read only normalized_landmarks_v1; all_xyz_v1 is never required or opened."""

import json
from pathlib import Path

from face_recognition.domain.interfaces.normalization_evaluation import ConditionKey
from face_recognition.domain.models.dataset import CONDITIONS, validate_identifier, validate_raw_landmarks
from face_recognition.domain.models.face_landmarks import FaceLandmarks, Landmark
from face_recognition.domain.models.processed_sample import (
    NORMALIZED_REPRESENTATION, PROCESSED_SCHEMA_VERSION, NormalizedLandmarkSample, SampleProvenance,
)


class NormalizedDatasetReadError(ValueError):
    """Safe paths/categories only; original decoder exceptions are suppressed."""


def _reject_constant(value: str) -> None:
    raise ValueError("Non-finite JSON constants are not allowed")


def normalized_from_json(line: str) -> NormalizedLandmarkSample:
    try:
        data = json.loads(line, parse_constant=_reject_constant)
        if (not isinstance(data, dict) or type(data.get("schema_version")) is not int
                or data["schema_version"] != PROCESSED_SCHEMA_VERSION or data["representation"] != NORMALIZED_REPRESENTATION):
            raise ValueError
        provenance = SampleProvenance(**{name: data[name] for name in SampleProvenance.__dataclass_fields__})
        if data["source_relative_path"] != provenance.source_relative_path:
            raise ValueError
        count, points = data["landmark_count"], data["normalized_landmarks"]
        if (type(count) is not int or count <= 0 or not isinstance(points, list) or len(points) != count
                or any(not isinstance(point, list) or len(point) != 3 for point in points)):
            raise ValueError
        landmarks = FaceLandmarks(tuple(Landmark(*point) for point in points))
        # Reuse the existing space-independent numeric/count validation only.
        validate_raw_landmarks(landmarks, count)
        return NormalizedLandmarkSample(provenance, landmarks)
    except Exception:
        raise NormalizedDatasetReadError("Invalid normalized sample JSON/schema") from None


class NormalizedDatasetReader:
    def __init__(self, processed_root: Path) -> None:
        self.processed_root = Path(processed_root).resolve()
        self.root = self.processed_root / NORMALIZED_REPRESENTATION
        self._checked(self.root)
        if not self.root.is_dir():
            raise NormalizedDatasetReadError(f"Missing normalized representation directory: {self.root}")
        self._manifest = self._read_manifest()
        self.expected_sample_count = self._manifest["total_samples_processed"]

    def _checked(self, path: Path) -> Path:
        try:
            relative = path.relative_to(self.processed_root)
            if ".." in relative.parts or not path.resolve().is_relative_to(self.processed_root):
                raise ValueError
            current = self.processed_root
            for part in relative.parts:
                current /= part
                if current.is_symlink():
                    raise ValueError
        except (ValueError, OSError, RuntimeError):
            raise NormalizedDatasetReadError(f"Normalized paths must remain inside the processed root without symlinks: {path}") from None
        return path

    def _read_manifest(self) -> dict:
        path = self._checked(self.root / "processing_manifest.json")
        try:
            with path.open("r", encoding="utf-8") as stream:
                data = json.load(stream, parse_constant=_reject_constant)
            if (not isinstance(data, dict) or type(data.get("schema_version")) is not int
                    or data["schema_version"] != PROCESSED_SCHEMA_VERSION
                    or data["representation"] != NORMALIZED_REPRESENTATION or data["completion_state"] != "completed"):
                raise ValueError
            for field in ("participants", "sessions", "conditions", "condition_files", "total_samples_processed",
                          "rejected_samples", "error_count"):
                if type(data[field]) is not int or data[field] < 0:
                    raise ValueError
            if data["rejected_samples"] or data["error_count"]:
                raise ValueError
            return data
        except Exception:
            raise NormalizedDatasetReadError(f"Invalid or missing completed normalized manifest: {path}") from None

    def _entries(self, directory: Path) -> tuple[Path, ...]:
        try:
            return tuple(sorted(directory.iterdir(), key=lambda path: path.name))
        except OSError:
            raise NormalizedDatasetReadError(f"Cannot list normalized directory: {directory}") from None

    def discover_conditions(self) -> tuple[ConditionKey, ...]:
        keys = []
        for person in self._entries(self.root):
            self._checked(person)
            if person.name == "processing_manifest.json" and person.is_file():
                continue
            try:
                validate_identifier(person.name)
                if not person.is_dir():
                    raise ValueError
            except ValueError:
                raise NormalizedDatasetReadError(f"Invalid normalized participant directory: {person}") from None
            for session in self._entries(person):
                self._checked(session)
                try:
                    validate_identifier(session.name)
                    if not session.is_dir():
                        raise ValueError
                except ValueError:
                    raise NormalizedDatasetReadError(f"Invalid normalized session directory: {session}") from None
                for path in self._entries(session):
                    self._checked(path)
                    if not path.is_file() or path.suffix != ".jsonl" or path.stem not in CONDITIONS:
                        raise NormalizedDatasetReadError(f"Unexpected normalized condition file: {path}")
                    keys.append((person.name, session.name, path.stem))
        keys.sort(key=lambda key: (key[0], key[1], CONDITIONS.index(key[2])))
        if (len(keys) != self._manifest["conditions"] or len(keys) != self._manifest["condition_files"]
                or len({key[0] for key in keys}) != self._manifest["participants"]
                or len({key[:2] for key in keys}) != self._manifest["sessions"]):
            raise NormalizedDatasetReadError("Normalized file catalog does not match its processing manifest counts")
        return tuple(keys)

    def read_condition(self, key: ConditionKey) -> tuple[NormalizedLandmarkSample, ...]:
        person, session, condition = key
        try:
            validate_identifier(person)
            validate_identifier(session)
            if condition not in CONDITIONS:
                raise ValueError
        except ValueError:
            raise NormalizedDatasetReadError("Invalid normalized condition key") from None
        if self._read_manifest() != self._manifest:
            raise NormalizedDatasetReadError("Normalized manifest changed during evaluation")
        path = self._checked(self.root / person / session / (condition + ".jsonl"))
        try:
            with path.open("r", encoding="utf-8") as stream:
                samples = tuple(normalized_from_json(line) for line in stream)
            if not samples:
                raise ValueError
            for index, sample in enumerate(samples):
                provenance = sample.provenance
                if (provenance.person_id, provenance.session_id, provenance.condition) != key or provenance.sample_index != index:
                    raise ValueError
            return samples
        except Exception:
            raise NormalizedDatasetReadError(f"Invalid normalized condition samples/metadata: {path}") from None
