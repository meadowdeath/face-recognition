"""Atomic publication of external, aggregate-only baseline reports."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile

from face_recognition.config.settings import PROJECT_ROOT
from face_recognition.domain.models.baseline_evaluation import BaselineEvaluationReport


class BaselineReportError(ValueError):
    pass


def validate_baseline_report_target(path: Path, input_roots: tuple[Path, ...]) -> Path:
    if Path(path).is_symlink():
        raise BaselineReportError("Baseline report target must not be a symlink")
    target = Path(path).resolve()
    if target.is_relative_to(PROJECT_ROOT.resolve()):
        raise BaselineReportError("Baseline reports must remain outside the Git repository")
    for root in input_roots:
        source = Path(root).resolve()
        if target.is_relative_to(source) or source.is_relative_to(target):
            raise BaselineReportError("Baseline report must not overlap its processed or split inputs")
    if target.exists():
        raise BaselineReportError("Baseline report already exists; choose a new output path")
    return target


def baseline_report_to_json(report: BaselineEvaluationReport) -> str:
    if report.schema_version != 1 or report.completion_state != "completed":
        raise BaselineReportError("Only completed baseline reports may be published")
    data = asdict(report)
    for classifier in data["classifiers"]:
        configuration = classifier["configuration"]
        configuration["classifier_parameters"] = dict(configuration["classifier_parameters"])
        if configuration["scaler_parameters"] is not None:
            configuration["scaler_parameters"] = dict(configuration["scaler_parameters"])
    try:
        return json.dumps(data, allow_nan=False, indent=2) + "\n"
    except (ValueError, TypeError):
        raise BaselineReportError("Invalid aggregate baseline report structure") from None


def write_baseline_report(report: BaselineEvaluationReport, path: Path, input_roots: tuple[Path, ...]) -> None:
    target = validate_baseline_report_target(path, input_roots)
    payload = baseline_report_to_json(report)
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
        raise BaselineReportError("Unable to publish baseline report without overwrite") from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
