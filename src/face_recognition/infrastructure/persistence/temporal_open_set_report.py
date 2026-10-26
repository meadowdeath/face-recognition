"""External aggregate temporal reports; no per-frame or per-window sequences."""

from dataclasses import asdict
import json
from pathlib import Path

from face_recognition.domain.models.temporal_open_set import TEMPORAL_MODEL, TEMPORAL_SCORES, WINDOW_SIZES, TemporalOpenSetReport
from face_recognition.infrastructure.persistence.baseline_evaluation_report import BaselineReportError, write_aggregate_report_json


def temporal_report_to_json(report: TemporalOpenSetReport) -> str:
    expected = tuple((family, size) for family in TEMPORAL_SCORES for size in WINDOW_SIZES)
    if (report.schema_version != 1 or report.completion_state != "completed" or report.model != TEMPORAL_MODEL
            or report.window_sizes != WINDOW_SIZES
            or tuple((row.score_family, row.window_size) for row in report.results) != expected
            or len(report.ranking) != len(expected) or set(report.ranking) != {row.identifier for row in report.results}
            or report.best_temporal_open_set_development_candidate != report.ranking[0]
            or report.known_test_payloads_accessed is not False or report.unknown_holdout_payloads_accessed is not False
            or any(row.vs_window1 is None or len(row.folds) != 2 for row in report.results)):
        raise BaselineReportError("Only completed fixed-model, protected temporal reports may be published")
    data = asdict(report)
    data["configuration"]["classifier_parameters"] = dict(data["configuration"]["classifier_parameters"])
    data["configuration"]["scaler_parameters"] = dict(data["configuration"]["scaler_parameters"])
    data["score_definitions"] = dict(data["score_definitions"])
    for row, original in zip(data["results"], report.results):
        row["configuration_id"] = original.identifier
    try:
        return json.dumps(data, allow_nan=False, indent=2) + "\n"
    except (ValueError, TypeError):
        raise BaselineReportError("Invalid aggregate temporal open-set report") from None


def write_temporal_report(report: TemporalOpenSetReport, path: Path, inputs: tuple[Path, ...]) -> None:
    write_aggregate_report_json(temporal_report_to_json(report), path, inputs)
