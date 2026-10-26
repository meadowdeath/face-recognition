"""Atomic external V2 artifacts; load trusted local joblib files only."""

from dataclasses import asdict, dataclass, field
from math import isfinite
import os
from pathlib import Path
import tempfile
import json

import joblib
import numpy as np
import sklearn
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted

from face_recognition.application.refit_v2_open_set_model import threshold_stability
from face_recognition.domain.models.dataset import validate_identifier
from face_recognition.domain.models.open_set_model import OpenSetModelError
from face_recognition.domain.models.v2_open_set_model import V2TransferGuard
from face_recognition.infrastructure.classifiers.sklearn_v2_open_set_refit import FittedV2Parts
from face_recognition.infrastructure.persistence.model_repository import _hash, _read_json, _write_json, validate_model_target
from face_recognition.infrastructure.persistence.v2_open_set_config_repository import frozen_v2_configuration_to_json

V2_ARTIFACTS = ("feature_selector.joblib", "scaler.joblib", "identity_svm.joblib", "threshold.json")


def _threshold_payload(manifest, snapshot):
    a = snapshot["architecture"]
    return {"schema_version": 1, "frozen_config_sha256": manifest["frozen_config_sha256"],
            "threshold": manifest["threshold"], "rejection_score": a["rejection_score_family"],
            "threshold_type": a["rejection_threshold_type"], "score_orientation": "higher_is_known",
            "temporal_aggregation": a["rejection_temporal_aggregation"], "window_size": a["temporal_window_size"],
            "calibration_method": "cross_session_oof", "selection_rule": snapshot["final_threshold_calibration_policy"]["selection_rule"],
            "policy": snapshot["final_threshold_calibration_policy"], "analysis": manifest["calibration_analysis"],
            "fold_threshold_stability": manifest["threshold_stability"],
            "known_oof_window_count": manifest["known_oof_counts"]["windows"],
            "unknown_fold_window_observation_count": manifest["unknown_fold_observation_counts"]["windows"]}


@dataclass(frozen=True, eq=False)
class LoadedV2OpenSetModel:
    parts: FittedV2Parts = field(repr=False)
    manifest: dict = field(repr=False)
    threshold: float


def _validate_metadata(manifest, config, config_sha256):
    snapshot = json.loads(frozen_v2_configuration_to_json(config))
    a, e = config.architecture, config.development_evidence
    if (not isinstance(config_sha256, str) or len(config_sha256) != 64 or any(c not in "0123456789abcdef" for c in config_sha256)
            or manifest["schema_version"] != 1 or manifest["model_id"] != "open_set_v2"
            or manifest["status"] != "fitted_and_calibrated" or manifest["completion_state"] != "completed"
            or manifest["frozen_config_sha256"] != config_sha256 or manifest["frozen_configuration"] != snapshot
            or manifest["frozen_configuration_id"] != config.configuration_id
            or manifest["diagnostic_report_sha256"] != config.provenance.sha256
            or manifest["source_identity_split_sha256"] != config.provenance.original_identity_split_sha256
            or manifest["architecture"] != snapshot["architecture"] or manifest["future_final_test"] != snapshot["future_final_test"]
            or manifest["source_representation"] != a.source_representation or manifest["original_feature_count"] != a.original_feature_count
            or manifest["selected_feature_count"] != a.selected_features or manifest["known_training_samples"] != e.known_source_samples
            or manifest["unknown_unique_source_samples"] != e.unknown_source_samples or manifest["identity_count"] != e.known_identity_count
            or manifest["calibration_method"] != "cross_session_oof" or manifest["v2_final_evaluation"] != "not_performed"
            or manifest["future_known_session_payloads_accessed"] is not False or manifest["future_unknown_payloads_accessed"] is not False
            or manifest["versions"]["scikit_learn"] != sklearn.__version__ or set(manifest["artifacts"]) != set(V2_ARTIFACTS)):
        raise ValueError
    numeric = manifest["threshold"]
    if type(numeric) not in (int, float) or not isfinite(numeric) or numeric != manifest["calibration_analysis"]["calibration"]["threshold"]:
        raise ValueError
    if (manifest["rejection_score"] != "svm_top1" or manifest["threshold_type"] != "global"
            or manifest["score_orientation"] != "higher_is_known" or manifest["window_size"] != 7
            or manifest["temporal_aggregation"] != "median" or manifest["identity_aggregation"] != "majority_vote"
            or manifest["identity_consistency_gate"] is not False):
        raise ValueError
    transfer = manifest["transfer_validation"]
    if (transfer["status"] != "passed" or transfer["guard"] != asdict(V2TransferGuard())
            or transfer["score_unit"] < V2TransferGuard().minimum_score_unit
            or not 0 <= transfer["maximum_shift_units"] <= V2TransferGuard().failure_units
            or not 0 <= transfer["fold_threshold_range_units"] <= V2TransferGuard().failure_units):
        raise ValueError
    sessions = sorted(f.calibration_session for f in config.known_cross_session_folds)
    if manifest["known_session_ids"] != sessions or manifest["fold_count"] != 3 or len(manifest["folds"]) != 3:
        raise ValueError
    for key in ("known_oof_counts", "unknown_unique_counts", "unknown_fold_observation_counts"):
        counts = manifest[key]
        if (any(type(counts[k]) is not int or counts[k] < 0 for k in ("source_samples", "used_samples", "discarded_samples", "windows"))
                or counts["windows"] <= 0 or counts["windows"] * a.temporal_window_size != counts["used_samples"]
                or counts["used_samples"] + counts["discarded_samples"] != counts["source_samples"]):
            raise ValueError
    if (manifest["known_oof_counts"]["source_samples"] != e.known_source_samples
            or manifest["known_oof_counts"]["windows"] != e.known_unique_windows
            or manifest["unknown_unique_counts"]["source_samples"] != e.unknown_source_samples
            or manifest["unknown_fold_observation_counts"]["windows"] != e.unknown_three_model_window_observations
            or any(manifest["unknown_fold_observation_counts"][k] != 3 * v for k, v in manifest["unknown_unique_counts"].items())
            or manifest["calibration_analysis"]["known_distribution"]["count"] != manifest["known_oof_counts"]["windows"]
            or manifest["calibration_analysis"]["unknown_distribution"]["count"] != manifest["unknown_fold_observation_counts"]["windows"]):
        raise ValueError
    if (transfer["oof_known"] != manifest["calibration_analysis"]["known_distribution"]
            or transfer["oof_unknown_fold_observations"] != manifest["calibration_analysis"]["unknown_distribution"]
            or transfer["final_model_known_development"]["count"] != manifest["known_oof_counts"]["windows"]
            or transfer["final_model_unknown_development"]["count"] != manifest["unknown_unique_counts"]["windows"]):
        raise ValueError
    for fold, expected in zip(manifest["folds"], snapshot["known_cross_session_folds"]):
        k = fold["known_counts"]
        if (fold["protocol"] != expected or fold["unknown_counts"] != manifest["unknown_unique_counts"]
                or fold["fit_samples"] != e.known_source_samples - k["source_samples"]
                or k["used_samples"] != k["windows"] * a.temporal_window_size
                or k["used_samples"] + k["discarded_samples"] != k["source_samples"]
                or fold["analysis"]["known_distribution"]["count"] != k["windows"]
                or fold["analysis"]["unknown_distribution"]["count"] != manifest["unknown_unique_counts"]["windows"]):
            raise ValueError
    if any(sum(f["known_counts"][k] for f in manifest["folds"]) != value for k, value in manifest["known_oof_counts"].items()):
        raise ValueError
    stable = threshold_stability(f["analysis"]["calibration"]["threshold"] for f in manifest["folds"])
    if manifest["threshold_stability"] != json.loads(json.dumps(asdict(stable))):
        raise ValueError
    return snapshot


def _load_artifacts(root, manifest, config, config_sha256):
    snapshot = _validate_metadata(manifest, config, config_sha256)
    for name in V2_ARTIFACTS:
        path = root / name
        if path.is_symlink() or not path.is_file() or _hash(path) != manifest["artifacts"][name]["sha256"]:
            raise ValueError
    if _read_json(root / "threshold.json") != _threshold_payload(manifest, snapshot):
        raise ValueError
    # Every hash and policy is checked BEFORE any pickle is deserialized.
    selector, scaler, svm = (joblib.load(root / name) for name in V2_ARTIFACTS[:3])
    if not isinstance(selector, SelectKBest) or not isinstance(scaler, StandardScaler) or not isinstance(svm, SVC):
        raise ValueError
    for part in (selector, scaler, svm):
        check_is_fitted(part)
    a, n = config.architecture, manifest["known_training_samples"]
    if (selector.k != a.selected_features or selector.score_func is not f_classif or selector.n_features_in_ != a.original_feature_count
            or selector.scores_.shape != (a.original_feature_count,) or selector.get_support().sum() != a.selected_features
            or scaler.n_features_in_ != a.selected_features or svm.n_features_in_ != a.selected_features
            or any(scaler.get_params()[key] != v for key, v in a.scaler_parameters)
            or any(svm.get_params()[key] != v for key, v in a.svm_parameters)
            or svm.decision_function_shape != a.decision_function_shape or svm.break_ties != a.svc_break_ties or svm.fit_status_ != 0
            or svm._gamma != dict(a.svm_parameters)["gamma"] or svm.shape_fit_ != (n, a.selected_features)
            or scaler.n_samples_seen_ != n):
        raise ValueError
    for vector in (scaler.mean_, scaler.var_, scaler.scale_):
        if vector.shape != (a.selected_features,) or not np.isfinite(vector).all():
            raise ValueError
    if np.any(scaler.scale_ <= 0) or np.any(scaler.var_ < 0):
        raise ValueError
    ids = tuple(manifest["class_ids"])
    for person in ids:
        validate_identifier(person)
    count = len(ids)
    if (ids != tuple(sorted(set(ids))) or count < 2 or count != manifest["identity_count"]
            or not np.array_equal(svm.classes_, np.arange(count)) or len(svm.n_support_) != count
            or int(svm.n_support_.sum()) != len(svm.support_) or len(svm.support_) != manifest["support_vector_count"]
            or not 0 < len(svm.support_) <= n or svm.support_vectors_.shape != (len(svm.support_), a.selected_features)
            or not all(np.isfinite(v).all() for v in (svm.support_vectors_, svm.dual_coef_, svm.intercept_))):
        raise ValueError
    return LoadedV2OpenSetModel(FittedV2Parts(selector, scaler, svm, ids), manifest, float(manifest["threshold"]))


def load_v2_open_set_model(path, config, config_sha256):
    """Integrity/shape/policy validation only: no fit, transform, predict or dataset reads."""
    try:
        root = Path(path).resolve()
        if (Path(path).is_symlink() or (root / "manifest.json").is_symlink()
                or {p.name for p in root.iterdir()} != {"manifest.json", *V2_ARTIFACTS}):
            raise ValueError
        return _load_artifacts(root, _read_json(root / "manifest.json"), config, config_sha256)
    except Exception:
        raise OpenSetModelError("Incomplete, incompatible or corrupted frozen V2 model artifacts") from None


def persist_v2_open_set_model(result, config, config_sha256, output, inputs):
    target = validate_model_target(output, inputs)
    stage = None
    try:
        snapshot = json.loads(frozen_v2_configuration_to_json(config))
        parts, summary = result.parts, result.summary
        if not isinstance(parts, FittedV2Parts) or summary.frozen_config_sha256 != config_sha256:
            raise ValueError
        target.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent))
        for name, item in zip(V2_ARTIFACTS[:3], (parts.selector, parts.scaler, parts.svm)):
            joblib.dump(item, stage / name, compress=3)
        manifest = {**asdict(summary), "schema_version": 1, "model_id": "open_set_v2", "status": "fitted_and_calibrated",
                    "completion_state": "completed", "frozen_configuration_id": config.configuration_id,
                    "frozen_configuration": snapshot, "diagnostic_report_sha256": config.provenance.sha256,
                    "source_representation": config.architecture.source_representation, "architecture": snapshot["architecture"],
                    "future_final_test": snapshot["future_final_test"], "class_ids": parts.class_ids,
                    "threshold": summary.analysis.calibration.threshold, "calibration_analysis": asdict(summary.analysis),
                    "rejection_score": "svm_top1", "threshold_type": "global", "score_orientation": "higher_is_known",
                    "window_size": 7, "temporal_aggregation": "median", "identity_aggregation": "majority_vote", "identity_consistency_gate": False,
                    "artifact_privacy": "Biometric-derived trusted local-only joblib files; pseudonymous class IDs are inference mapping only; no source sample IDs or vectors in metadata"}
        manifest.pop("analysis")
        manifest = json.loads(json.dumps(manifest, allow_nan=False))
        _write_json(stage / "threshold.json", _threshold_payload(manifest, snapshot))
        for name in V2_ARTIFACTS:
            with (stage / name).open("r+b") as stream:
                os.fsync(stream.fileno())
        manifest["artifacts"] = {name: {"sha256": _hash(stage / name)} for name in V2_ARTIFACTS}
        _load_artifacts(stage, manifest, config, config_sha256)
        _write_json(stage / "manifest.json", manifest)  # Completion marker is the LAST artifact written.
        validate_model_target(target, inputs)
        os.rename(stage, target)
    except BaseException as error:
        if stage is not None and stage.exists():
            try:
                (stage / "manifest.json").unlink(missing_ok=True)
                _write_json(stage / "INCOMPLETE.json", {"status": "incomplete", "usable": False})
            except Exception:
                pass
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        raise OpenSetModelError("Unable to publish V2 model; any retained staging directory is incomplete and must not be used") from None
