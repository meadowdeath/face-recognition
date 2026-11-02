"""Immutable selection provenance and aggregate same-model calibration reports."""

from dataclasses import asdict
import json

from face_recognition.application.create_v3_cv_protocol import metadata_sha256
from face_recognition.application.evaluate_v3_mahalanobis_calibration import validate_selection_provenance
from face_recognition.domain.models.rejection import RejectionError
from face_recognition.domain.models.v3_mahalanobis_calibration import (
    METHOD, THRESHOLD_TRANSPORT, SelectionReportProvenance, V3MahalanobisConfiguration,
)
from face_recognition.domain.models.v3_novelty_evaluation import CANDIDATES, V3NoveltyConfiguration
from face_recognition.infrastructure.persistence.aggregate_report import write_aggregate_report_json
from face_recognition.infrastructure.persistence.artifact_integrity import file_sha256
from face_recognition.infrastructure.persistence.v3_cv_protocol_repository import _read_json


def read_mahalanobis_selection_report(path, protocol):
    """Read initial report for method/provenance only, never threshold/metric inputs."""
    try:
        before = file_sha256(path)
        source = _read_json(path)
        if (type(source["schema_version"]) is not int or source["schema_version"] != 1
                or source["experiment"] != "v3_nested_novelty_comparison_v1" or source["completion_state"] != "completed"
                or metadata_sha256(source["configuration"]) != metadata_sha256(V3NoveltyConfiguration())
                or source["processing_manifest_sha256"] != protocol.catalog.processing_manifest_sha256
                or type(source["identity_count"]) is not int or type(source["source_samples"]) is not int
                or len(source["outer_results"]) != 4 or len(source["selection_frequency"]) != 4
                or any(type(count) is not int for _, count in source["selection_frequency"])
                or tuple((name, count) for name, count in source["selection_frequency"])
                   != tuple((name, 4 if name == METHOD else 0) for name in CANDIDATES)):
            raise ValueError
        selected = tuple((r["fold_name"], r["selected_method"]) for r in source["outer_results"])
        warnings = tuple(r["transport"]["warning"] for r in source["outer_results"])
        if any(type(value) is not bool for value in warnings) or file_sha256(path) != before:
            raise ValueError
        result = SelectionReportProvenance(before, source["protocol_sha256"], source["catalog_sha256"],
                    source["payload_sha256"], source["configuration_sha256"], source["identity_count"],
                    source["source_samples"], selected, sum(warnings))
        validate_selection_provenance(result, protocol)
        return result
    except Exception:
        raise RejectionError("Invalid selection report: protocol/configuration mismatch or Mahalanobis not selected in all folds") from None


def mahalanobis_calibration_report_to_json(report):
    if (type(report.schema_version) is not int or report.schema_version != 1
            or report.experiment != "v3_mahalanobis_same_model_calibration_v1" or report.completion_state != "completed"
            or metadata_sha256(report.configuration) != metadata_sha256(V3MahalanobisConfiguration())
            or report.configuration_sha256 != metadata_sha256(report.configuration)
            or report.threshold_transport != THRESHOLD_TRANSPORT or len(report.outer_results) != 4
            or any(r.threshold_transport != THRESHOLD_TRANSPORT or r.fit_calls != 1
                   or not r.same_model_identity_verified or not r.no_refit_after_calibration
                   or r.model_sha256_before_calibration != r.model_sha256_after_test for r in report.outer_results)):
        raise RejectionError("Incomplete or inconsistent same-model calibration report")
    return json.dumps(asdict(report), sort_keys=True, allow_nan=False, indent=2) + "\n"


def write_mahalanobis_calibration_report(report, output, inputs):
    write_aggregate_report_json(mahalanobis_calibration_report_to_json(report), output, inputs)
