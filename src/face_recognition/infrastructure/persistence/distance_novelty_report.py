"""External aggregate-only distance novelty report, atomic and no-overwrite."""

from dataclasses import asdict
import json
from pathlib import Path

from face_recognition.domain.models.distance_novelty import NOVELTY_MODEL, NOVELTY_NEIGHBORS, REFERENCE_SCORES, DistanceNoveltyReport
from face_recognition.infrastructure.persistence.baseline_evaluation_report import BaselineReportError, write_aggregate_report_json


def distance_novelty_report_to_json(report: DistanceNoveltyReport) -> str:
    expected = tuple(name for name, _ in NOVELTY_NEIGHBORS)
    all_results = report.novelty_results + report.reference_results
    if (report.schema_version != 1 or report.completion_state != "completed" or report.model != NOVELTY_MODEL
            or tuple((row.score_family, row.neighbor_count) for row in report.novelty_results) != NOVELTY_NEIGHBORS
            or tuple(row.score_family for row in report.reference_results) != REFERENCE_SCORES
            or any(row.neighbor_count is not None for row in report.reference_results)
            or len(report.ranking) != 3 or set(report.ranking) != set(expected)
            or report.best_distance_based_novelty_development_candidate != report.ranking[0]
            or len(report.neighbor_diagnostics) != 2
            or any(len(row.folds) != 2 for row in all_results)
            or any(tuple(ref.reference_family for ref in row.vs_references) != REFERENCE_SCORES for row in report.novelty_results)
            or any(row.known_distance_distribution is None or row.unknown_distance_distribution is None
                   or any(fold.known_distance_distribution is None or fold.unknown_distance_distribution is None for fold in row.folds)
                   for row in report.novelty_results)
            or report.known_test_payloads_accessed is not False or report.unknown_holdout_payloads_accessed is not False):
        raise BaselineReportError("Only complete, fixed-model, protected distance novelty reports may be published")
    data = asdict(report)
    data["configuration"]["classifier_parameters"] = dict(data["configuration"]["classifier_parameters"])
    data["configuration"]["scaler_parameters"] = dict(data["configuration"]["scaler_parameters"])
    data["score_definitions"] = dict(data["score_definitions"])
    try:
        return json.dumps(data, allow_nan=False, indent=2) + "\n"
    except (ValueError, TypeError):
        raise BaselineReportError("Invalid aggregate distance novelty report") from None


def write_distance_novelty_report(report: DistanceNoveltyReport, path: Path, inputs: tuple[Path, ...]) -> None:
    write_aggregate_report_json(distance_novelty_report_to_json(report), path, inputs)
