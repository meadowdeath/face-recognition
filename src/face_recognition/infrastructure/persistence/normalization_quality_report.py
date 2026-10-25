"""Write aggregate-only evaluation results outside both input trees."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile

from face_recognition.application.evaluate_normalization import NormalizationQualityReport


def write_quality_report(report: NormalizationQualityReport, path: Path, input_roots: tuple[Path, ...]) -> None:
    if Path(path).is_symlink():
        raise ValueError("Report target must not be a symlink")
    target = Path(path).resolve()
    for root in input_roots:
        resolved = Path(root).resolve()
        if target.is_relative_to(resolved) or resolved.is_relative_to(target):
            raise ValueError("Aggregate report must be outside raw and processed input trees")
    if target.exists():
        raise ValueError("Report target already exists; choose a new report path")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(asdict(report), allow_nan=False, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)  # Atomic no-overwrite publication, as used by raw capture.
    finally:
        temporary.unlink(missing_ok=True)
