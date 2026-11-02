"""Paired representation authorization and aggregate, immutable publication."""

from dataclasses import asdict, replace
import json
from pathlib import Path

from face_recognition.application.create_v3_cv_protocol import metadata_sha256
from face_recognition.application.evaluate_v3_pose_representation import rank_representations, validate_representation_pair
from face_recognition.domain.models.dataset import CONDITIONS
from face_recognition.domain.models.processed_sample import BASELINE_REPRESENTATION, POSE_REPRESENTATION
from face_recognition.domain.models.rejection import RejectionError
from face_recognition.domain.models.v3_mahalanobis_calibration import V3MahalanobisConfiguration
from face_recognition.domain.models.v3_pose_representation import POSE_DELTA_CONDITIONS
from face_recognition.infrastructure.features.pose_landmark_normalizer import POSE_NORMALIZATION
from face_recognition.infrastructure.persistence.aggregate_report import write_aggregate_report_json
from face_recognition.infrastructure.persistence.artifact_integrity import file_sha256
from face_recognition.infrastructure.persistence.v3_cv_protocol_repository import _read_json, read_v3_catalog
from face_recognition.infrastructure.persistence.v3_payload_loader import load_v3_payloads


def pose_processing_fingerprints(root):
    """Require the explicitly versioned geometric transform before feature reads."""
    if {p.name for p in Path(root).iterdir()} != {"processing_manifest.json", POSE_REPRESENTATION}:
        raise RejectionError("Pose output must contain only its declared representation and run manifest")
    paths = (Path(root) / "processing_manifest.json", Path(root) / POSE_REPRESENTATION / "processing_manifest.json")
    before = tuple(file_sha256(p) for p in paths)
    overall, representation = (_read_json(p) for p in paths)
    for row in (overall, representation):
        if (row.get("schema_version") != 1 or row.get("completion_state") != "completed"
                or row.get("normalization") != "PoseLandmarkNormalizer"
                or row.get("normalization_parameters") != POSE_NORMALIZATION
                or row.get("feature_extraction") != "LandmarkFeatureExtractor"
                or row.get("inference_dimensions") != "per_source_sample"
            or row.get("error_count") != 0 or row.get("rejected_samples") != 0):
            raise RejectionError("Pose processing manifest does not declare the expected transform")
    for field in ("participants", "sessions", "conditions", "total_samples_processed", "rejected_samples", "error_count"):
        if type(overall.get(field)) is not int or overall[field] != representation.get(field):
            raise RejectionError("Pose processing manifests disagree")
    if (overall.get("representations") != [POSE_REPRESENTATION] or representation.get("representation") != POSE_REPRESENTATION
            or overall.get("condition_files_per_representation") != representation.get("condition_files")
            or overall.get("total_condition_files") != representation.get("condition_files")
            or tuple(file_sha256(p) for p in paths) != before):
        raise RejectionError("Inconsistent or changing pose processing manifest")
    return before


def load_paired_pose_payloads(baseline_root, candidate_root, protocol):
    first, second = Path(baseline_root).resolve(), Path(candidate_root).resolve()
    if first.is_relative_to(second) or second.is_relative_to(first):
        raise RejectionError("Baseline and candidate roots must be separate")
    manifests = pose_processing_fingerprints(candidate_root)
    baseline = load_v3_payloads(baseline_root, protocol)
    candidate = load_v3_payloads(candidate_root, protocol, representation=POSE_REPRESENTATION)
    validate_representation_pair(baseline, candidate)
    catalog = read_v3_catalog(candidate_root, representation=POSE_REPRESENTATION)
    if manifests != pose_processing_fingerprints(candidate_root):
        raise RejectionError("Pose processing metadata changed while loading")
    return baseline, candidate, catalog, manifests


def pose_representation_report_to_json(report):
    if (report.schema_version != 1 or report.completion_state != "completed"
            or report.experiment != "v3_pose_representation_comparison_v1"
            or report.threshold_transport != "none_same_model_calibration"
            or tuple(r.representation for r in report.results) != (BASELINE_REPRESENTATION, POSE_REPRESENTATION)
            or report.ranking != rank_representations(report.results) or report.selected_development_representation != report.ranking[0]
            or report.normalization_definition != POSE_NORMALIZATION
            or report.normalization_definition_sha256 != metadata_sha256(POSE_NORMALIZATION)
            or tuple(r.condition for r in report.condition_comparison) != CONDITIONS
            or tuple(name for name, _ in report.pose_distance_frr_deltas) != POSE_DELTA_CONDITIONS):
        raise RejectionError("Incomplete controlled representation comparison")
    for result in report.results:
        if (len(result.folds) != 4 or result.configuration != replace(V3MahalanobisConfiguration(), representation=result.representation)
                or result.configuration.window_size != 7
                or result.configuration.novelty_method != "mahalanobis_distance"
                or result.configuration_sha256 != metadata_sha256(result.configuration)
                or any(f.evaluation.fit_calls != 1 or not f.evaluation.no_refit_after_calibration
                       or not f.evaluation.same_model_identity_verified
                       or f.evaluation.model_sha256_before_calibration != f.evaluation.model_sha256_after_test
                       or tuple(r.label for r in f.known_per_condition) != CONDITIONS
                       or tuple(r.label for r in f.unknown_per_condition) != CONDITIONS for f in result.folds)):
            raise RejectionError("Incomplete or refitted representation fold")
    if tuple(f.evaluation.roles for f in report.results[0].folds) != tuple(f.evaluation.roles for f in report.results[1].folds):
        raise RejectionError("Representation fold roles differ")
    return json.dumps(asdict(report), sort_keys=True, allow_nan=False, indent=2) + "\n"


def write_pose_representation_report(report, output, inputs):
    write_aggregate_report_json(pose_representation_report_to_json(report), output, inputs)
