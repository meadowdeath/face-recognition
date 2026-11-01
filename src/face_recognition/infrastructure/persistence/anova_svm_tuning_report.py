"""External, aggregate tuning reports; no recognizer/transformer serialization."""

from dataclasses import asdict
import json
from pathlib import Path

from face_recognition.domain.models.anova_svm_tuning import (
    ANOVA_K_VALUES, REFERENCE_CANDIDATE, AnovaSvmSearchSpace, AnovaSvmTuningReport, tuning_candidates,
)
from face_recognition.infrastructure.persistence.baseline_evaluation_report import BaselineReportError, write_aggregate_report_json


def anova_svm_report_to_json(report: AnovaSvmTuningReport) -> str:
    expected_ids = {row.identifier for row in tuning_candidates()}
    if (report.schema_version != 1 or report.completion_state != "completed"
            or tuple(row.candidate for row in report.results) != tuning_candidates()
            or report.configurations_evaluated != len(report.results) or report.folds_per_configuration != 2
            or report.search_space != AnovaSvmSearchSpace() or report.reference_candidate != REFERENCE_CANDIDATE
            or tuple(row.rank for row in report.ranking) != tuple(range(1, len(report.results) + 1))
            or {row.candidate_id for row in report.ranking} != expected_ids
            or report.selected_development_candidate.identifier != report.ranking[0].candidate_id
            or tuple(row.k for row in report.selection_stability) != ANOVA_K_VALUES
            or any(row.vs_full_svm_baseline is None or row.vs_anova512_reference is None for row in report.results)):
        raise BaselineReportError("Only complete fixed-grid ANOVA-SVM tuning reports may be published")
    data = asdict(report)
    evaluations = [data["full_feature_svm_reference"], *(row["evaluation"] for row in data["results"])]
    for evaluation in evaluations:
        configuration = evaluation["configuration"]
        configuration["classifier_parameters"] = dict(configuration["classifier_parameters"])
        configuration["scaler_parameters"] = dict(configuration["scaler_parameters"])
    for row, original in zip(data["results"], report.results):
        row["candidate_id"] = original.candidate.identifier
    data["selected_development_candidate"]["candidate_id"] = report.selected_development_candidate.identifier
    try:
        return json.dumps(data, allow_nan=False, indent=2) + "\n"
    except (ValueError, TypeError):
        raise BaselineReportError("Invalid aggregate ANOVA-SVM tuning report") from None


def write_anova_svm_report(report: AnovaSvmTuningReport, path: Path, input_roots: tuple[Path, ...]) -> None:
    write_aggregate_report_json(anova_svm_report_to_json(report), path, input_roots)
