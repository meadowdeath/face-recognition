"""Aggregate immutable final report; model files are hashed, never written."""

from dataclasses import asdict
import json
from pathlib import Path

from face_recognition.application.final_evaluation_data import FinalEvaluationError
from face_recognition.domain.models.final_evaluation import CALIBRATION_WARNING, NO_RETUNING, FinalEvaluationReport
from face_recognition.infrastructure.persistence.baseline_evaluation_report import write_aggregate_report_json
from face_recognition.infrastructure.persistence.model_repository import model_fingerprints


def final_evaluation_report_to_json(report: FinalEvaluationReport) -> str:
    if (report.schema_version != 1 or report.status != "final_holdout_evaluation_completed"
            or report.no_retuning != NO_RETUNING or report.calibration_warning != CALIBRATION_WARNING
            or report.known_test_payloads_accessed is not True or report.unknown_holdout_payloads_accessed is not True
            or report.model_artifacts_unchanged is not True or report.fitting_performed is not False
            or report.threshold_calibration_performed is not False
            or not report.frame_novelty_diagnostics.diagnostic_only or not report.temporal_novelty_diagnostics.diagnostic_only):
        raise FinalEvaluationError("Only complete, immutable-model, no-refit final reports may be published")
    data = asdict(report)
    data["context"]["model_fingerprints"] = dict(data["context"]["model_fingerprints"])
    data["context"]["model_manifest_sha256"] = data["context"]["model_fingerprints"]["manifest.json"]
    data["context"]["configuration"]["architecture"]["svm_parameters"] = dict(data["context"]["configuration"]["architecture"]["svm_parameters"])
    data["context"]["configuration"]["architecture"]["scaler_parameters"] = dict(data["context"]["configuration"]["architecture"]["scaler_parameters"])
    try:
        return json.dumps(data, allow_nan=False, sort_keys=True, indent=2) + "\n"
    except (ValueError, TypeError):
        raise FinalEvaluationError("Invalid aggregate final report") from None


def write_final_evaluation_report(report: FinalEvaluationReport, output: Path, inputs: tuple[Path, ...], model_dir: Path) -> None:
    if model_fingerprints(model_dir) != report.context.model_fingerprints:
        raise FinalEvaluationError("Fitted model files changed during final evaluation; report publication refused")
    write_aggregate_report_json(final_evaluation_report_to_json(report), output, inputs)
