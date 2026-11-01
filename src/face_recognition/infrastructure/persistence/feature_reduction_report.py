"""Serialize reduction aggregates only, using existing safe report publication."""

from dataclasses import asdict
import json
from pathlib import Path

from face_recognition.domain.models.feature_reduction import FeatureReductionReport
from face_recognition.infrastructure.persistence.baseline_evaluation_report import BaselineReportError, write_aggregate_report_json


def feature_reduction_report_to_json(report: FeatureReductionReport) -> str:
    if report.schema_version != 1 or report.completion_state != "completed":
        raise BaselineReportError("Only completed feature reduction reports may be published")
    data = asdict(report)
    for result in data["results"]:
        configuration = result["evaluation"]["configuration"]
        configuration["classifier_parameters"] = dict(configuration["classifier_parameters"])
        if configuration["scaler_parameters"] is not None:
            configuration["scaler_parameters"] = dict(configuration["scaler_parameters"])
    try:
        return json.dumps(data, allow_nan=False, indent=2) + "\n"
    except (ValueError, TypeError):
        raise BaselineReportError("Invalid aggregate feature reduction report") from None


def write_feature_reduction_report(report: FeatureReductionReport, path: Path, input_roots: tuple[Path, ...]) -> None:
    write_aggregate_report_json(feature_reduction_report_to_json(report), path, input_roots)
