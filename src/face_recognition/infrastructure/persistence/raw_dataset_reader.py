"""Strictly read-only access to the versioned raw landmark dataset.

No capture/resume entry point is called. Participants and sessions sort by name;
conditions follow the manifest's completed protocol order, and samples retain
their validated zero-based file order. Incomplete sessions may be inspected,
but only their declared completed conditions are readable. Known .tmp files
are left untouched; uncommitted final files are errors, never recovered here.
"""

from collections.abc import Iterator
from dataclasses import dataclass
import json
from pathlib import Path

from face_recognition.domain.models.dataset import (
    CONDITIONS, RawLandmarkSample, SessionManifest, validate_identifier,
)
from face_recognition.infrastructure.persistence.dataset_repository import (
    FilesystemDatasetRepository, manifest_from_dict,
)


class RawDatasetReadError(ValueError):
    """Invalid source structure/data; messages contain paths, never sample values."""


@dataclass(frozen=True)
class RawDatasetSession:
    directory: Path
    manifest: SessionManifest


def _reject_manifest_constant(value: str) -> None:
    raise ValueError("Non-finite JSON constants are not allowed")


class RawDatasetReader:
    """Discover sessions and load validated raw samples from a caller-owned root.

    Reading does not write files, recover capture state, or set timestamps.
    Filesystem-managed access-time updates may still occur. The source should
    remain unchanged during iteration; this is not a concurrent capture reader.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise RawDatasetReadError(f"Raw dataset root must be an existing directory: {self.root}")
        # Only read_condition() is used on this adapter, never open_session().
        self._repository = FilesystemDatasetRepository(self.root)

    def _checked_path(self, path: Path) -> Path:
        path = path if path.is_absolute() else self.root / path
        try:
            relative = path.relative_to(self.root)
            if ".." in relative.parts or not path.resolve().is_relative_to(self.root):
                raise ValueError
            current = self.root
            for part in relative.parts:
                current /= part
                if current.is_symlink():
                    raise ValueError
        except (ValueError, OSError, RuntimeError):
            raise RawDatasetReadError(f"Dataset paths must stay inside the root without symlinks: {path}") from None
        return path

    def _children(self, directory: Path) -> tuple[Path, ...]:
        try:
            return tuple(sorted(directory.iterdir(), key=lambda entry: entry.name))
        except OSError:
            raise RawDatasetReadError(f"Cannot list dataset directory: {directory}") from None

    def _identifier_directory(self, directory: Path) -> None:
        self._checked_path(directory)
        try:
            validate_identifier(directory.name)
        except ValueError:
            raise RawDatasetReadError(f"Invalid participant/session directory identifier: {directory}") from None
        if not directory.is_dir():
            raise RawDatasetReadError(f"Expected participant/session directory: {directory}")

    def read_manifest(self, session_directory: Path) -> SessionManifest:
        """Read a session directory (absolute, or relative to root) without recovery."""
        directory = self._checked_path(Path(session_directory))
        relative = directory.relative_to(self.root)
        if len(relative.parts) != 2:
            raise RawDatasetReadError(f"Expected <person>/<session> beneath the raw root: {directory}")
        self._identifier_directory(directory.parent)
        self._identifier_directory(directory)
        path = self._checked_path(directory / "manifest.json")
        if not path.is_file():
            raise RawDatasetReadError(f"Missing expected manifest.json file: {path}")
        try:
            with path.open("r", encoding="utf-8") as stream:
                data = json.load(stream, parse_constant=_reject_manifest_constant)
            if not isinstance(data, dict):
                raise ValueError
            manifest = manifest_from_dict(data)
        except (ValueError, TypeError, OSError, OverflowError):
            # Do not expose decoder exceptions which could embed source values.
            raise RawDatasetReadError(f"Invalid session manifest JSON/schema: {path}") from None
        if manifest.config.person_id != directory.parent.name:
            raise RawDatasetReadError(f"Manifest person_id does not match its directory: {path}")
        if manifest.config.session_id != directory.name:
            raise RawDatasetReadError(f"Manifest session_id does not match its directory: {path}")
        return manifest

    def _validate_session_files(self, directory: Path, manifest: SessionManifest) -> None:
        completed = {condition + ".jsonl" for condition in manifest.completed_conditions}
        temporary = {condition + ".jsonl.tmp" for condition in CONDITIONS} | {"manifest.json.tmp"}
        for entry in self._children(directory):
            self._checked_path(entry)
            if not entry.is_file():
                raise RawDatasetReadError(f"Unexpected non-file entry in session: {entry}")
            if entry.name in completed or entry.name in temporary or entry.name in ("manifest.json", ".gitkeep"):
                continue
            if entry.suffix == ".jsonl" and entry.stem in CONDITIONS:
                raise RawDatasetReadError(f"Uncommitted condition file; capture recovery is required separately: {entry}")
            raise RawDatasetReadError(f"Unexpected file in dataset session: {entry}")
        for name in sorted(completed):
            path = self._checked_path(directory / name)
            if not path.is_file():
                raise RawDatasetReadError(f"Missing expected completed condition file: {path}")

    def discover_sessions(self) -> tuple[RawDatasetSession, ...]:
        """Discover manifests and declared files; sample contents load separately."""
        sessions = []
        for person in self._children(self.root):
            if person.name == ".gitkeep" and person.is_file():
                self._checked_path(person)
                continue
            self._identifier_directory(person)
            for directory in self._children(person):
                if directory.name == ".gitkeep" and directory.is_file():
                    self._checked_path(directory)
                    continue
                manifest = self.read_manifest(directory)
                self._validate_session_files(directory, manifest)
                sessions.append(RawDatasetSession(directory, manifest))
        return tuple(sessions)

    def read_condition(self, session: RawDatasetSession, condition: str) -> tuple[RawLandmarkSample, ...]:
        """Validate a complete condition before returning any of its samples."""
        directory = self._checked_path(session.directory)
        manifest = self.read_manifest(directory)
        if manifest != session.manifest:
            raise RawDatasetReadError(f"Session manifest changed since discovery: {session.directory}")
        self._validate_session_files(directory, manifest)
        if condition not in manifest.completed_conditions:
            raise RawDatasetReadError(f"Condition is not declared completed in manifest: {session.directory}")
        try:
            return self._repository.read_condition(manifest, condition)
        except (ValueError, TypeError, OSError, OverflowError):
            path = session.directory / (condition + ".jsonl")
            raise RawDatasetReadError(
                f"Invalid condition data: check metadata, indices, dimensions, landmarks, counts and timestamps: {path}"
            ) from None

    def iter_samples(self, session: RawDatasetSession | None = None) -> Iterator[RawLandmarkSample]:
        """Iterate one session, or all sessions, retaining identity/session metadata."""
        sessions = self.discover_sessions() if session is None else (session,)
        for current in sessions:
            # Validate even an empty/incomplete session supplied explicitly.
            directory = self._checked_path(current.directory)
            manifest = self.read_manifest(directory)
            if manifest != current.manifest:
                raise RawDatasetReadError(f"Session manifest changed since discovery: {current.directory}")
            self._validate_session_files(directory, manifest)
            for condition in manifest.completed_conditions:
                yield from self.read_condition(current, condition)
