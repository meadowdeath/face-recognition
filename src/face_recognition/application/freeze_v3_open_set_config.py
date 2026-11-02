"""Validate already completed aggregate evidence and seal decisions; no fitting."""

from dataclasses import asdict, fields, replace
from math import isfinite
from pathlib import PurePosixPath, PureWindowsPath
import re

from face_recognition.application.create_v3_cv_protocol import metadata_sha256, validate_v3_cv_protocol
from face_recognition.application.evaluate_v3_mahalanobis_calibration import cyclic_session_roles
from face_recognition.domain.models.classification import ClassificationMetrics
from face_recognition.domain.models.dataset import CONDITIONS
from face_recognition.domain.models.processed_sample import BASELINE_REPRESENTATION, POSE_REPRESENTATION
from face_recognition.domain.models.rejection import RejectionError, RejectionMetrics
from face_recognition.domain.models.v3_mahalanobis_calibration import V3MahalanobisCalibrationReport, V3MahalanobisConfiguration
from face_recognition.domain.models.v3_mahalanobis_diagnostics import MahalanobisAggregateDiagnostics, V3MahalanobisDiagnosticsReport
from face_recognition.domain.models.v3_novelty_evaluation import CANDIDATES, V3NoveltyConfiguration, V3NoveltyReport
from face_recognition.domain.models.v3_open_set_config import (
    EVIDENCE_KINDS, EVIDENCE_SOURCE_TYPES, IMPLEMENTATION_MODULES, DevelopmentMetricEvidence, FrozenV3OpenSetConfiguration,
)
from face_recognition.domain.models.v3_pose_representation import RepresentationResult, V3PoseRepresentationReport


def _object(row, model):
    if not isinstance(row, dict) or set(row) != {f.name for f in fields(model)}:
        raise RejectionError("Incomplete or unexpected aggregate evidence fields")
    return row


def _digest(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise RejectionError("Invalid SHA-256 provenance")


def _rate(value):
    if type(value) not in (int, float) or not isfinite(value) or not 0 <= value <= 1:
        raise RejectionError("Evidence metrics must be finite rates")
    return float(value)


def _configuration(row, expected):
    if (metadata_sha256(row["configuration"]) != metadata_sha256(expected)
            or row["configuration_sha256"] != metadata_sha256(expected)):
        raise RejectionError("Evidence pipeline differs from the completed V3 configuration")


def _same_model_fold(row, roles):
    if (row["roles"] != asdict(roles)  # JSON arrays and dataclass tuples normalize below.
            and metadata_sha256(row["roles"]) != metadata_sha256(roles)):
        raise RejectionError("Evidence session/identity roles do not match the protocol")
    if (type(row["fit_calls"]) is not int or row["fit_calls"] != 1
            or row["no_refit_after_calibration"] is not True or row["same_model_identity_verified"] is not True
            or row["threshold_transport"] != "none_same_model_calibration"
            or row["model_sha256_before_calibration"] != row["model_sha256_after_test"]
            or type(row["threshold"]) not in (int, float) or not isfinite(row["threshold"])):
        raise RejectionError("Evidence does not preserve same-model calibration")
    _digest(row["model_sha256_before_calibration"])


def _summary(row):
    result = {}
    for group, model in (("mean_closed_set", ClassificationMetrics), ("mean_metrics", RejectionMetrics)):
        for name, value in _object(row[group], model).items():
            result[group + "." + name] = _rate(value)
    for name in ("mean_roc_auc", "mean_eer"):
        result[name] = _rate(row[name])
    return result


def freeze_v3_open_set_configuration(protocol, documents, references, normalization, implementations):
    """Only metadata in; curated immutable configuration out. Never open a catalog."""
    validate_v3_cv_protocol(protocol)
    if tuple(documents) != EVIDENCE_KINDS[1:] or tuple(r.kind for r in references) != EVIDENCE_KINDS:
        raise RejectionError("The complete ordered V3 evidence chain is required")
    refs = {r.kind: r for r in references}
    models = (V3NoveltyReport, V3MahalanobisCalibrationReport, V3MahalanobisDiagnosticsReport, V3PoseRepresentationReport)
    names = ("v3_nested_novelty_comparison_v1", "v3_mahalanobis_same_model_calibration_v1",
             "v3_mahalanobis_diagnostics_v1", "v3_pose_representation_comparison_v1")
    for (kind, row), model, name in zip(documents.items(), models, names):
        _object(row, model)
        if (type(row["schema_version"]) is not int or row["schema_version"] != 1
                or row["experiment"] != name or row["completion_state"] != "completed"
                or refs[kind].source_type != name):
            raise RejectionError("Incomplete or incompatible development evidence")
    novelty, calibration, diagnostics, pose = (documents[k] for k in EVIDENCE_KINDS[1:])
    roles = cyclic_session_roles(protocol)
    for row in (novelty, calibration, pose):
        if row["protocol_sha256"] != protocol.protocol_sha256 or row["protocol_file_sha256"] != refs["protocol"].sha256:
            raise RejectionError("Development protocol/file provenance mismatch")
    for row in (novelty, calibration):
        if (row["catalog_sha256"] != protocol.catalog_sha256
                or row["processing_manifest_sha256"] != protocol.catalog.processing_manifest_sha256
                or row["identity_count"] != len(protocol.specification.known_ids)
                or row["source_samples"] != protocol.catalog.declared_source_samples):
            raise RejectionError("Development population metadata mismatch")
    _configuration(novelty, V3NoveltyConfiguration())
    _configuration(calibration, V3MahalanobisConfiguration())
    if (len(novelty["outer_results"]) != 4
            or tuple((r["fold_name"], r["selected_method"]) for r in novelty["outer_results"])
            != tuple((r.fold_name, "mahalanobis_distance") for r in roles)
            or tuple(tuple(pair) for pair in novelty["selection_frequency"])
            != tuple((name, 4 if name == "mahalanobis_distance" else 0) for name in CANDIDATES)):
        raise RejectionError("Novelty evidence did not select Mahalanobis in all folds")
    selection = calibration["selection_provenance"]
    if (selection["report_sha256"] != refs["novelty_report"].sha256
            or selection["payload_sha256"] != novelty["payload_sha256"]
            or calibration["payload_sha256"] != novelty["payload_sha256"]):
        raise RejectionError("Calibration does not reference the supplied novelty evidence")
    if calibration["threshold_transport"] != "none_same_model_calibration" or len(calibration["outer_results"]) != 4:
        raise RejectionError("Missing same-model calibration evidence")
    for row, role in zip(calibration["outer_results"], roles):
        _same_model_fold(row, role)
    provenance = diagnostics["provenance"]
    expected_provenance = {
        "protocol_sha256": protocol.protocol_sha256, "protocol_file_sha256": refs["protocol"].sha256,
        "evaluation_report_sha256": refs["calibration_report"].sha256,
        "initial_selection_report_sha256": refs["novelty_report"].sha256,
        "catalog_sha256": protocol.catalog_sha256, "processing_manifest_sha256": calibration["processing_manifest_sha256"],
        "payload_sha256": calibration["payload_sha256"], "configuration_sha256": calibration["configuration_sha256"],
    }
    if (diagnostics["method"] != "mahalanobis_distance" or diagnostics["threshold_transport"] != "none_same_model_calibration"
            or any(provenance.get(k) != v for k, v in expected_provenance.items()) or len(diagnostics["folds"]) != 4):
        raise RejectionError("Diagnostics do not reference the supplied study")
    for row, saved in zip(diagnostics["folds"], calibration["outer_results"]):
        if (row["fold_name"] != saved["roles"]["fold_name"] or row["saved_threshold"] != saved["threshold"]
                or row["saved_model_state_verified"] is not True or row["saved_outer_summaries_verified"] is not True
                or row["known_counts"] != saved["known_test_counts"] or row["unknown_counts"] != saved["unknown_test_counts"]):
            raise RejectionError("Unverified or incompatible diagnostic replay")
    aggregate = _object(diagnostics["aggregate"], MahalanobisAggregateDiagnostics)
    for field, rate_name in (("known_condition_severity", "frr"), ("unknown_condition_severity", "far")):
        rows = aggregate[field]
        if len(rows) != len(CONDITIONS) or {r["label"] for r in rows} != set(CONDITIONS):
            raise RejectionError("Incomplete diagnostic condition evidence")
        for row in rows: _rate(row["counts"][rate_name])
    for field, labels in (("known_session_severity", protocol.specification.known_sessions),
                          ("unknown_identity_severity", protocol.specification.unknown_ids)):
        if len(aggregate[field]) != len(labels) or {r["label"] for r in aggregate[field]} != set(labels):
            raise RejectionError("Incomplete diagnostic session/identity evidence")
    if (pose["baseline_protocol_catalog_sha256"] != protocol.catalog_sha256
            or pose["threshold_transport"] != "none_same_model_calibration"
            or pose["normalization_definition_sha256"] != metadata_sha256(normalization)
            or metadata_sha256(pose["normalization_definition"]) != metadata_sha256(normalization)
            or tuple(r["representation"] for r in pose["results"]) != (BASELINE_REPRESENTATION, POSE_REPRESENTATION)
            or pose["selected_development_representation"] != POSE_REPRESENTATION
            or tuple(pose["ranking"]) != (POSE_REPRESENTATION, BASELINE_REPRESENTATION)):
        raise RejectionError("Pose evidence does not declare the required selected representation/transform")
    summaries, metrics = [], []
    for result in pose["results"]:
        _object(result, RepresentationResult)
        representation = result["representation"]
        _configuration(result, replace(V3MahalanobisConfiguration(), representation=representation))
        for key in ("payload_sha256", "catalog_sha256", "processing_manifest_sha256"):
            _digest(result[key])
        if representation == BASELINE_REPRESENTATION and any(result[key] != calibration[key] for key in ("payload_sha256", "catalog_sha256", "processing_manifest_sha256")):
            raise RejectionError("Pose baseline population differs from the same-model study")
        if len(result["folds"]) != 4:
            raise RejectionError("Pose evidence requires all four folds")
        for fold, role in zip(result["folds"], roles):
            _same_model_fold(fold["evaluation"], role)
        summary = _summary(result)
        summaries.append(summary)
        metrics.extend(DevelopmentMetricEvidence("pose_report", representation + "." + name, value) for name, value in sorted(summary.items()))
    # Check consistency of the published winner; never change it, fit models,
    # enumerate strategies or select operational thresholds in the freeze step.
    def published_key(summary):
        return (-summary["mean_metrics.binary_balanced_accuracy"], summary["mean_metrics.known_rejection_rate"],
                summary["mean_metrics.unknown_false_acceptance_rate"], -summary["mean_closed_set.macro_f1"])
    if published_key(summaries[1]) >= published_key(summaries[0]):
        raise RejectionError("Declared pose winner conflicts with the predeclared ranking (baseline wins exact ties)")
    if tuple(r["condition"] for r in pose["condition_comparison"]) != CONDITIONS:
        raise RejectionError("All conditions including pitch_down are required")
    for row in pose["condition_comparison"]:
        for key in ("baseline_frr", "candidate_frr"):
            metrics.append(DevelopmentMetricEvidence("pose_report", row["condition"] + "." + key, _rate(row[key])))
        if abs(row["frr_delta_candidate_minus_baseline"] - (row["candidate_frr"] - row["baseline_frr"])) > 1e-12:
            raise RejectionError("Inconsistent pose condition evidence")
    config = FrozenV3OpenSetConfiguration(normalization, metadata_sha256(normalization), tuple(implementations), tuple(references),
        protocol.protocol_sha256, protocol.catalog_sha256, tuple(metrics))
    config = replace(config, configuration_sha256=metadata_sha256(config))
    validate_frozen_v3_configuration(config)
    return config


def validate_frozen_v3_configuration(config):
    expected = FrozenV3OpenSetConfiguration(config.normalization, config.normalization_sha256, config.implementations,
        config.evidence, config.protocol_sha256, config.protocol_catalog_sha256, config.development_evidence)
    if (metadata_sha256(replace(config, configuration_sha256="")) != metadata_sha256(expected)
            or config.configuration_sha256 != metadata_sha256(expected)
            or config.normalization_sha256 != metadata_sha256(config.normalization)):
        raise RejectionError("Frozen configuration fields/digest are incompatible")
    for values in (config.evidence, config.implementations, config.development_evidence,
                   config.normalization.first_eye, config.normalization.second_eye,
                   config.normalization.superior_reference, config.normalization.inferior_reference):
        if not isinstance(values, tuple):
            raise RejectionError("Frozen metadata must use immutable sequences")
    if tuple(r.kind for r in config.evidence) != EVIDENCE_KINDS or tuple(r.module for r in config.implementations) != IMPLEMENTATION_MODULES:
        raise RejectionError("Missing evidence/implementation fingerprints")
    for row, source_type in zip(config.evidence, EVIDENCE_SOURCE_TYPES):
        _digest(row.sha256)
        name = row.filename
        if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+\.json", name)
                or PureWindowsPath(name).name != name or PurePosixPath(name).name != name or type(row.schema_version) is not int
                or row.schema_version != 1 or row.source_type != source_type):
            raise RejectionError("Evidence references must be technical JSON filenames, not paths")
    for row in config.implementations: _digest(row.sha256)
    for value in (config.protocol_sha256, config.protocol_catalog_sha256, config.normalization_sha256): _digest(value)
    if not config.development_evidence or len({(r.source, r.metric) for r in config.development_evidence}) != len(config.development_evidence):
        raise RejectionError("Missing or duplicate aggregate evidence metrics")
    for row in config.development_evidence:
        if row.source != "pose_report" or not re.fullmatch(r"[a-z0-9_.]+", row.metric):
            raise RejectionError("Unsupported evidence metric")
        _rate(row.value)
