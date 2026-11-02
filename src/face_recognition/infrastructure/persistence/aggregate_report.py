"""Atomic no-overwrite publication of external aggregate JSON reports."""

import os
from pathlib import Path
import tempfile

from face_recognition.config.settings import PROJECT_ROOT


class AggregateReportError(ValueError):
    pass


def validate_aggregate_report_target(path: Path, input_roots: tuple[Path, ...]) -> Path:
    if Path(path).is_symlink():
        raise AggregateReportError("Aggregate report target must not be a symlink")
    target = Path(path).resolve()
    if target.is_relative_to(PROJECT_ROOT.resolve()):
        raise AggregateReportError("Aggregate reports must remain outside the Git repository")
    for root in input_roots:
        source = Path(root).resolve()
        if target.is_relative_to(source) or source.is_relative_to(target):
            raise AggregateReportError("Aggregate report must not overlap its processed or split inputs")
    if target.exists():
        raise AggregateReportError("Aggregate report already exists; choose a new output path")
    return target


def write_aggregate_report_json(payload: str, path: Path, input_roots: tuple[Path, ...]) -> None:
    """Shared atomic no-overwrite publication; payload must be aggregate JSON."""
    target = validate_aggregate_report_target(path, input_roots)
    temporary = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
        temporary = Path(name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)  # No-overwrite atomic publication on NTFS/ext4.
    except Exception:
        raise AggregateReportError("Unable to publish aggregate report without overwrite") from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
