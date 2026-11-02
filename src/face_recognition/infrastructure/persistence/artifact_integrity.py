"""Read-only artifact SHA-256 integrity; hashes are not authenticity signatures."""

from pathlib import Path
from hashlib import sha256

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
