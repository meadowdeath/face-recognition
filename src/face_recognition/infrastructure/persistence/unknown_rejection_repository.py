"""External metadata-only unknown split and aggregate calibration reports."""

from dataclasses import asdict
import json
from pathlib import Path

from face_recognition.application.unknown_rejection_split import UnknownRejectionError, plan_unknown_split, validate_unknown_split
from face_recognition.domain.models.identity_split import IdentitySplitManifest
from face_recognition.domain.models.unknown_rejection import REJECTION_CANDIDATES, UnknownRejectionReport, UnknownRejectionSplit, UnknownSplitSpec
from face_recognition.infrastructure.persistence.baseline_evaluation_report import BaselineReportError, write_aggregate_report_json
from face_recognition.infrastructure.persistence.processed_dataset_catalog import reject_json_constant


def unknown_split_to_json(split: UnknownRejectionSplit) -> str:
    return json.dumps(asdict(split), allow_nan=False, indent=2) + "\n"


def write_unknown_split(split: UnknownRejectionSplit, source: IdentitySplitManifest, output: Path, source_path: Path) -> None:
    validate_unknown_split(source, split)
    write_aggregate_report_json(unknown_split_to_json(split), output, (source_path,))


def _unique(pairs):
    value = dict(pairs)
    if len(value) != len(pairs):
        raise ValueError
    return value


def read_unknown_split(path: Path, source: IdentitySplitManifest) -> UnknownRejectionSplit:
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            data = json.load(stream, parse_constant=reject_json_constant, object_pairs_hook=_unique)
        spec = UnknownSplitSpec(tuple(data["specification"]["calibration_ids"]), tuple(data["specification"]["holdout_ids"]))
        expected = plan_unknown_split(source, spec)
        # Full schema, array order, types, counts and source binding must match.
        if json.dumps(data, sort_keys=True) != json.dumps(json.loads(unknown_split_to_json(expected)), sort_keys=True):
            raise ValueError
        return expected
    except Exception:
        raise UnknownRejectionError(f"Invalid or unreadable unknown rejection split: {Path(path)}") from None


def rejection_report_to_json(report: UnknownRejectionReport) -> str:
    ids = {candidate.identifier for candidate in REJECTION_CANDIDATES}
    if (report.schema_version != 1 or report.completion_state != "completed"
            or tuple(row.candidate for row in report.results) != REJECTION_CANDIDATES
            or len(report.ranking) != len(ids) or set(report.ranking) != ids
            or report.selected_open_set_development_candidate.identifier != report.ranking[0]
            or report.protected_payloads_accessed is not False):
        raise BaselineReportError("Only completed two-candidate rejection calibration reports may be published")
    data = asdict(report)
    for row in data["results"]:
        configuration = row["configuration"]
        configuration["classifier_parameters"] = dict(configuration["classifier_parameters"])
        configuration["scaler_parameters"] = dict(configuration["scaler_parameters"])
    try:
        return json.dumps(data, allow_nan=False, indent=2) + "\n"
    except (ValueError, TypeError):
        raise BaselineReportError("Invalid aggregate rejection calibration report") from None


def write_rejection_report(report: UnknownRejectionReport, output: Path, inputs: tuple[Path, ...]) -> None:
    write_aggregate_report_json(rejection_report_to_json(report), output, inputs)
