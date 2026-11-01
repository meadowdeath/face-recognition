"""Atomic, immutable, aggregate-only final V2 report; no model writes."""

from dataclasses import asdict
import json

from face_recognition.application.final_evaluation_data import FinalEvaluationError
from face_recognition.domain.models.final_v2_evaluation import V2_NO_RETUNING
from face_recognition.infrastructure.persistence.baseline_evaluation_report import write_aggregate_report_json
from face_recognition.infrastructure.persistence.v2_model_repository import read_v2_model_metadata


def final_v2_report_to_json(report):
    if (type(report.schema_version) is not int or report.schema_version != 1
            or report.experiment != "final_v2_open_set_evaluation_v1" or report.status != "final_holdout_evaluation_completed"
            or report.no_retuning != V2_NO_RETUNING or report.fitting_performed is not False
            or report.threshold_calibration_performed is not False or report.model_artifacts_unchanged is not True
            or report.final_known_payloads_accessed is not True or report.final_unknown_payloads_accessed is not True
            or not report.frame_score_diagnostics.diagnostic_only or not report.temporal_score_diagnostics.diagnostic_only):
        raise FinalEvaluationError("Only complete, no-refit final V2 reports may be published")
    data = asdict(report)
    data["closed_set_delta_final_minus_development"] = dict(data["closed_set_delta_final_minus_development"])
    data["context"]["model_fingerprints"] = dict(data["context"]["model_fingerprints"])
    for key in ("svm_parameters", "scaler_parameters"):
        data["context"]["configuration"]["architecture"][key] = dict(data["context"]["configuration"]["architecture"][key])
    try:
        return json.dumps(data, allow_nan=False, sort_keys=True, indent=2) + "\n"
    except (TypeError, ValueError):
        raise FinalEvaluationError("Invalid aggregate final V2 report") from None


def write_final_v2_report(report, output, inputs, model_dir):
    metadata, fingerprints = read_v2_model_metadata(model_dir, report.context.configuration, report.context.frozen_config_sha256)
    if fingerprints != report.context.model_fingerprints or metadata["threshold"] != report.context.threshold:
        raise FinalEvaluationError("V2 model artifacts changed; final report publication refused")
    write_aggregate_report_json(final_v2_report_to_json(report), output, inputs)
