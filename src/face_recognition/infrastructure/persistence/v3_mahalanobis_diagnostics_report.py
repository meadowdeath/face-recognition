"""Read immutable aggregate evidence; publish aggregate post-hoc diagnostics."""

from dataclasses import asdict
from math import isfinite
from statistics import fmean, pstdev
import json
import re

from face_recognition.application.create_v3_cv_protocol import metadata_sha256
from face_recognition.application.evaluate_v3_mahalanobis_calibration import cyclic_session_roles, validate_selection_provenance
from face_recognition.application.v3_novelty_calibration import moments
from face_recognition.domain.models.classification import ClassificationMetrics, EvaluationVersions
from face_recognition.domain.models.rejection import RejectionError, RejectionMetrics
from face_recognition.domain.models.score_analysis import EqualErrorPoint, ScoreDistribution, ScoreSeparation
from face_recognition.domain.models.temporal import WindowCounts
from face_recognition.domain.models.v3_mahalanobis_calibration import (
    THRESHOLD_TRANSPORT, CyclicFoldRoles, MahalanobisCalibrationFold, SelectionReportProvenance,
    V3MahalanobisCalibrationReport, V3MahalanobisConfiguration,
)
from face_recognition.domain.models.v3_novelty_evaluation import BinaryCalibrationMetrics
from face_recognition.infrastructure.persistence.aggregate_report import write_aggregate_report_json
from face_recognition.infrastructure.persistence.artifact_integrity import file_sha256
from face_recognition.infrastructure.persistence.v3_cv_protocol_repository import _object, _read_json


def _number(value, *, rate=False):
    if type(value) not in (int, float) or not isfinite(value) or (rate and not 0 <= value <= 1):
        raise ValueError
    return float(value)


def _rates(value, model):
    return model(**{key: _number(v, rate=True) for key, v in _object(value, model).items()})


def _counts(value):
    row = _object(value, WindowCounts)
    if (any(type(v) is not int or v < 0 for v in row.values()) or not row["windows"]
            or row["used_samples"] != row["windows"] * 7
            or row["used_samples"] + row["discarded_samples"] != row["source_samples"]):
        raise ValueError
    return WindowCounts(**row)


def _distribution(value, expected_count):
    row = _object(value, ScoreDistribution)
    if type(row["count"]) is not int or row["count"] != expected_count:
        raise ValueError
    for key in row.keys() - {"count"}:
        row[key] = _number(row[key])
    if row["std"] < 0 or not row["p10"] <= row["p25"] <= row["median"] <= row["p75"] <= row["p90"]:
        raise ValueError
    return ScoreDistribution(**row)


def _separation(value, known_count, unknown_count):
    row = _object(value, ScoreSeparation)
    if row["diagnostic_only"] is not True:
        raise ValueError
    row["roc_auc"] = _number(row["roc_auc"], rate=True)
    eer = _object(row["equal_error_point"], EqualErrorPoint)
    row["equal_error_point"] = EqualErrorPoint(**{k: _number(v, rate=k != "threshold") for k, v in eer.items()})
    row["known_distribution"] = _distribution(row["known_distribution"], known_count)
    row["unknown_distribution"] = _distribution(row["unknown_distribution"], unknown_count)
    return ScoreSeparation(**row)


def read_mahalanobis_evaluation_report(path, protocol, protocol_file_sha256):
    """Validate source structure/provenance without opening feature payloads."""
    try:
        before = file_sha256(path)
        row = _object(_read_json(path), V3MahalanobisCalibrationReport)
        config = V3MahalanobisConfiguration()
        if (type(row["schema_version"]) is not int or row["schema_version"] != 1
                or row["experiment"] != "v3_mahalanobis_same_model_calibration_v1" or row["completion_state"] != "completed"
                or row["threshold_transport"] != THRESHOLD_TRANSPORT
                or row["protocol_sha256"] != protocol.protocol_sha256 or row["protocol_file_sha256"] != protocol_file_sha256
                or row["catalog_sha256"] != protocol.catalog_sha256
                or row["processing_manifest_sha256"] != protocol.catalog.processing_manifest_sha256
                or metadata_sha256(row["configuration"]) != metadata_sha256(config)
                or row["configuration_sha256"] != metadata_sha256(config)
                or row["identity_count"] != len(protocol.specification.known_ids)
                or row["source_samples"] != protocol.catalog.declared_source_samples):
            raise ValueError
        row["configuration"] = config
        selection = _object(row["selection_provenance"], SelectionReportProvenance)
        selection["selected_methods"] = tuple(tuple(pair) for pair in selection["selected_methods"])
        row["selection_provenance"] = SelectionReportProvenance(**selection)
        validate_selection_provenance(row["selection_provenance"], protocol, row["payload_sha256"])
        row["versions"] = EvaluationVersions(**_object(row["versions"], EvaluationVersions))
        folds = []
        for raw in row["outer_results"]:
            fold = _object(raw, MahalanobisCalibrationFold)
            roles = _object(fold["roles"], CyclicFoldRoles)
            for name in ("known_fit_sessions", "unknown_calibration_ids", "unknown_test_ids"):
                roles[name] = tuple(roles[name])
            fold["roles"] = CyclicFoldRoles(**roles)
            if (fold["threshold_transport"] != THRESHOLD_TRANSPORT or type(fold["fit_calls"]) is not int or fold["fit_calls"] != 1
                    or fold["same_model_identity_verified"] is not True or fold["no_refit_after_calibration"] is not True
                    or fold["model_sha256_before_calibration"] != fold["model_sha256_after_test"]
                    or not re.fullmatch(r"[a-f0-9]{64}", fold["model_sha256_before_calibration"])
                    or type(fold["fit_known_frames"]) is not int or fold["fit_known_frames"] < 1):
                raise ValueError
            fold["threshold"] = _number(fold["threshold"])
            for name in ("known_calibration_counts", "unknown_calibration_counts", "known_test_counts", "unknown_test_counts"):
                fold[name] = _counts(fold[name])
            fold["calibration_metrics"] = _rates(fold["calibration_metrics"], BinaryCalibrationMetrics)
            fold["operational_metrics"] = _rates(fold["operational_metrics"], RejectionMetrics)
            fold["closed_set_known_frames"] = _rates(fold["closed_set_known_frames"], ClassificationMetrics)
            fold["calibration_diagnostics"] = _separation(fold["calibration_diagnostics"], fold["known_calibration_counts"].windows, fold["unknown_calibration_counts"].windows)
            fold["diagnostics"] = _separation(fold["diagnostics"], fold["known_test_counts"].windows, fold["unknown_test_counts"].windows)
            folds.append(MahalanobisCalibrationFold(**fold))
        if tuple(f.roles for f in folds) != cyclic_session_roles(protocol):
            raise ValueError
        row["outer_results"] = tuple(folds)
        for prefix, model, values in (("closed_set", ClassificationMetrics, [f.closed_set_known_frames for f in folds]),
                                       ("metrics", RejectionMetrics, [f.operational_metrics for f in folds])):
            mean, std = moments(values, model)
            for label, expected in (("mean_" + prefix, mean), ("std_" + prefix, std)):
                row[label] = _rates(row[label], model)
                if row[label] != expected: raise ValueError
        for name, values in (("roc_auc", [f.diagnostics.roc_auc for f in folds]), ("eer", [f.diagnostics.equal_error_point.eer for f in folds])):
            for prefix, operation in (("mean_", fmean), ("std_", pstdev)):
                row[prefix + name] = _number(row[prefix + name], rate=True)
                if row[prefix + name] != operation(values): raise ValueError
        for name, attr in (("far_range", "unknown_false_acceptance_rate"), ("frr_range", "known_rejection_rate")):
            values = [getattr(f.operational_metrics, attr) for f in folds]
            row[name] = tuple(_number(v, rate=True) for v in row[name])
            if row[name] != (min(values), max(values)): raise ValueError
        if file_sha256(path) != before:
            raise ValueError
        return V3MahalanobisCalibrationReport(**row), before
    except Exception:
        raise RejectionError("Invalid evaluation evidence: incomplete report, protocol/configuration mismatch or inconsistent saved summaries") from None


def mahalanobis_diagnostics_to_json(report):
    if (report.schema_version != 1 or report.completion_state != "completed" or len(report.folds) != 4
            or report.method != "mahalanobis_distance" or report.threshold_transport != THRESHOLD_TRANSPORT
            or any(not f.saved_model_state_verified or not f.saved_outer_summaries_verified for f in report.folds)):
        raise RejectionError("Only fully verified diagnostics may be published")
    return json.dumps(asdict(report), sort_keys=True, allow_nan=False, indent=2) + "\n"


def write_mahalanobis_diagnostics(report, output, inputs):
    write_aggregate_report_json(mahalanobis_diagnostics_to_json(report), output, inputs)
