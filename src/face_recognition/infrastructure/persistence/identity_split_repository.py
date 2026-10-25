"""Atomic, no-overwrite publication of external metadata-only split manifests."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile

from face_recognition.config.settings import PROJECT_ROOT
from face_recognition.domain.models.identity_split import IdentitySplitManifest


class IdentitySplitPersistenceError(ValueError):
    pass


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
