"""Shared read-only manifest/path/catalog validation for processed representations."""

import json
from pathlib import Path

from face_recognition.domain.interfaces.normalization_evaluation import ConditionKey
from face_recognition.domain.models.dataset import CONDITIONS, validate_identifier
from face_recognition.domain.models.processed_sample import PROCESSED_SCHEMA_VERSION, REPRESENTATIONS


class ProcessedDatasetReadError(ValueError):
    """Sanitized processed-source diagnostics."""


def reject_json_constant(value: str) -> None:
    raise ValueError("Non-finite JSON constants are not allowed")


class ProcessedDatasetCatalog:
    error_type = ProcessedDatasetReadError

    def __init__(self, processed_root: Path, representation: str) -> None:
        if representation not in REPRESENTATIONS:
            raise self.error_type("Unsupported processed representation")
        self.representation = representation
        self.processed_root = Path(processed_root).resolve()
        self.root = self.processed_root / representation
        self._checked(self.root)
        if not self.root.is_dir():
            raise self.error_type(f"Missing processed representation directory: {self.root}")
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
            raise self.error_type(f"Processed paths must remain inside the source root without symlinks: {path}") from None
        return path

    def _read_manifest(self) -> dict:
        path = self._checked(self.root / "processing_manifest.json")
        try:
            with path.open("r", encoding="utf-8") as stream:
                data = json.load(stream, parse_constant=reject_json_constant)
            if (not isinstance(data, dict) or type(data.get("schema_version")) is not int
                    or data["schema_version"] != PROCESSED_SCHEMA_VERSION
                    or data["representation"] != self.representation or data["completion_state"] != "completed"):
                raise ValueError
            for field in ("participants", "sessions", "conditions", "condition_files", "total_samples_processed",
                          "rejected_samples", "error_count"):
                if type(data[field]) is not int or data[field] < 0:
                    raise ValueError
            if data["rejected_samples"] or data["error_count"]:
                raise ValueError
            return data
        except Exception:
            raise self.error_type(f"Invalid or missing completed processed manifest: {path}") from None

    def _entries(self, directory: Path) -> tuple[Path, ...]:
        try:
            return tuple(sorted(directory.iterdir(), key=lambda path: path.name))
        except OSError:
            raise self.error_type(f"Cannot list processed directory: {directory}") from None

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
                raise self.error_type(f"Invalid processed participant directory: {person}") from None
            for session in self._entries(person):
                self._checked(session)
                try:
                    validate_identifier(session.name)
                    if not session.is_dir():
                        raise ValueError
                except ValueError:
                    raise self.error_type(f"Invalid processed session directory: {session}") from None
                for path in self._entries(session):
                    self._checked(path)
                    if not path.is_file() or path.suffix != ".jsonl" or path.stem not in CONDITIONS:
                        raise self.error_type(f"Unexpected processed condition file: {path}")
                    keys.append((person.name, session.name, path.stem))
        keys.sort(key=lambda key: (key[0], key[1], CONDITIONS.index(key[2])))
        if (len(keys) != self._manifest["conditions"] or len(keys) != self._manifest["condition_files"]
                or len({key[0] for key in keys}) != self._manifest["participants"]
                or len({key[:2] for key in keys}) != self._manifest["sessions"]):
            raise self.error_type("Processed file catalog does not match its processing manifest counts")
        return tuple(keys)
