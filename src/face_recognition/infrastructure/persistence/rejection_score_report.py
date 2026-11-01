"""Aggregate-only score comparison reports; no fitted score/model objects."""

from dataclasses import asdict
import json
from pathlib import Path

from face_recognition.domain.models.rejection_score_evaluation import SCORE_FAMILIES, RejectionScoreReport
from face_recognition.domain.models.unknown_rejection import REJECTION_CANDIDATES
from face_recognition.infrastructure.persistence.baseline_evaluation_report import BaselineReportError, write_aggregate_report_json


def rejection_score_report_to_json(report: RejectionScoreReport) -> str:
    expected = tuple((candidate, family) for candidate in REJECTION_CANDIDATES for family in SCORE_FAMILIES)
    ids = {row.identifier for row in report.results}
    if (report.schema_version != 1 or report.completion_state != "completed"
            or tuple((row.candidate, row.score_family) for row in report.results) != expected
            or len(report.ranking) != len(expected) or set(report.ranking) != ids
            or report.best_rejection_score_development_candidate != report.ranking[0]
            or report.known_test_payloads_accessed is not False or report.unknown_holdout_payloads_accessed is not False
            or any(row.vs_margin is None or len(row.folds) != 2 for row in report.results)):
        raise BaselineReportError("Only complete, protected eight-configuration score reports may be published")
    data = asdict(report)
    data["score_definitions"] = dict(data["score_definitions"])
    for row, original in zip(data["results"], report.results):
        row["configuration"]["classifier_parameters"] = dict(row["configuration"]["classifier_parameters"])
        row["configuration"]["scaler_parameters"] = dict(row["configuration"]["scaler_parameters"])
        row["configuration_id"] = original.identifier
    try:
        return json.dumps(data, allow_nan=False, indent=2) + "\n"
    except (TypeError, ValueError):
        raise BaselineReportError("Invalid aggregate rejection-score report") from None


def write_rejection_score_report(report: RejectionScoreReport, path: Path, inputs: tuple[Path, ...]) -> None:
    write_aggregate_report_json(rejection_score_report_to_json(report), path, inputs)
