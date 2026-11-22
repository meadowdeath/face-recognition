"""Read-only artifact SHA-256 integrity; hashes are not authenticity signatures."""

from pathlib import Path
from hashlib import sha256
from dataclasses import fields
import json

from face_recognition.config.settings import PROJECT_ROOT
from face_recognition.domain.models.rejection import RejectionError

def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_fingerprints(root: Path, filenames: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    """Hash explicitly named regular files without loading model parameters.

    Names are single relative filenames, unique and returned in lexical order.
    No version-specific required artifact set is implied.
    """
    root = Path(root)
    if root.is_symlink() or not root.is_dir() or not filenames or len(set(filenames)) != len(filenames):
        raise ValueError("Invalid artifact root or names")
    for name in filenames:
        if not isinstance(name, str) or name in ("", ".", "..") or "/" in name or "\\" in name or ":" in name:
            raise ValueError("Artifact names must be unique relative filenames")
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("Artifact must be an existing regular file")
    return tuple((name, file_sha256(root / name)) for name in sorted(filenames))


class ArtifactPublicationError(ValueError):
    pass


def validate_output_target(path: Path, input_roots: tuple[Path, ...]) -> Path:
    if Path(path).is_symlink():
        raise ArtifactPublicationError("Artifact target must not be a symlink")
    target = Path(path).resolve()
    if target.is_relative_to(PROJECT_ROOT.resolve()):
        raise ArtifactPublicationError("Artifacts must remain outside the Git repository")
    for root in input_roots:
        source = Path(root).resolve()
        if target.is_relative_to(source) or source.is_relative_to(target):
            raise ArtifactPublicationError("Artifact target must not overlap its inputs")
    if target.exists():
        raise ArtifactPublicationError("Artifact target already exists; choose a new output path")
    return target

def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise RejectionError("Duplicate JSON metadata key")
        result[key] = value
    return result


def _read_json(path):
    if Path(path).is_symlink(): raise RejectionError("Metadata source must not be a symlink")
    def reject(value): raise RejectionError("Non-finite JSON metadata")
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_object, parse_constant=reject)


def _object(value, model):
    if not isinstance(value, dict) or set(value) != {f.name for f in fields(model)}:
        raise RejectionError("Incomplete or unexpected metadata fields")
    return dict(value)


def publish_new_directory(stage: Path, target: Path) -> None:
    """Atomic adjacent-directory rename with no replacement, including empty targets."""
    import os
    from face_recognition.infrastructure.persistence.export_files import rename_exclusive

    if stage.parent != target.parent:
        raise ArtifactPublicationError("Publication requires sibling paths")
    if os.name == "nt":
        os.rename(stage, target)
    else:
        descriptor = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            rename_exclusive(descriptor, stage.name, target.name, target.parent)
        finally:
            os.close(descriptor)
