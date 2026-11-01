"""External aggregate temporal novelty reports with atomic no-overwrite writes."""

from dataclasses import asdict
import json
from pathlib import Path

from face_recognition.domain.models.distance_novelty import NOVELTY_MODEL
from face_recognition.domain.models.temporal_distance_novelty import TEMPORAL_DISTANCE_NEIGHBORS, TEMPORAL_DISTANCE_SCORES, TemporalDistanceNoveltyReport
from face_recognition.domain.models.temporal_open_set import WINDOW_SIZES
from face_recognition.infrastructure.persistence.baseline_evaluation_report import BaselineReportError, write_aggregate_report_json


def temporal_distance_report_to_json(report: TemporalDistanceNoveltyReport) -> str:
    expected = tuple((family, size) for family in TEMPORAL_DISTANCE_SCORES for size in WINDOW_SIZES)
    if (report.schema_version != 1 or report.completion_state != "completed" or report.model != NOVELTY_MODEL
            or report.window_sizes != WINDOW_SIZES or report.neighbor_counts != TEMPORAL_DISTANCE_NEIGHBORS
            or tuple((row.score_family, row.window_size) for row in report.results) != expected
            or len(report.ranking) != len(expected) or set(report.ranking) != {row.identifier for row in report.results}
            or report.best_temporal_distance_novelty_development_candidate != report.ranking[0]
            or any(row.vs_window1 is None or len(row.folds) != 2 for row in report.results)
            or report.known_test_payloads_accessed is not False or report.unknown_holdout_payloads_accessed is not False):
        raise BaselineReportError("Only completed, fixed-model, protected eight-configuration temporal novelty reports may be published")
    data = asdict(report)
    data["configuration"]["classifier_parameters"] = dict(data["configuration"]["classifier_parameters"])
    data["configuration"]["scaler_parameters"] = dict(data["configuration"]["scaler_parameters"])
    data["score_definitions"] = dict(data["score_definitions"])
    data["neighbor_counts"] = dict(data["neighbor_counts"])
    for row, original in zip(data["results"], report.results):
        row["configuration_id"] = original.identifier
    try:
        return json.dumps(data, allow_nan=False, indent=2) + "\n"
    except (ValueError, TypeError):
        raise BaselineReportError("Invalid aggregate temporal distance novelty report") from None


def write_temporal_distance_report(report: TemporalDistanceNoveltyReport, path: Path, inputs: tuple[Path, ...]) -> None:
    write_aggregate_report_json(temporal_distance_report_to_json(report), path, inputs)
