"""Trusted local fitted artifacts only; integrity hashes are not signatures."""

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import tempfile

import joblib
import numpy as np
import sklearn
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted

from face_recognition.application.refit_open_set_model import FittedOpenSetDevelopmentModel
from face_recognition.config.settings import PROJECT_ROOT
from face_recognition.domain.models.dataset import validate_identifier
from face_recognition.domain.models.open_set_model import OpenSetModelError
from face_recognition.infrastructure.classifiers.sklearn_open_set_refit import FittedOpenSetParts
from face_recognition.infrastructure.features.known_distances import KnownDistanceScorer
from face_recognition.infrastructure.persistence.open_set_config_repository import (
    LoadedFrozenConfiguration, _invalid_constant, _unique_object, frozen_configuration_to_json,
)

ARTIFACTS = ("feature_selector.joblib", "scaler.joblib", "identity_svm.joblib", "novelty_reference.npz", "threshold.json")


def validate_model_target(path: Path, inputs: tuple[Path, ...]) -> Path:
    target = Path(path).resolve()
    if Path(path).is_symlink() or target.is_relative_to(PROJECT_ROOT.resolve()):
        raise OpenSetModelError("Fitted model output must be external and must not be a symlink")
    if target.exists():
        raise OpenSetModelError("Model output already exists; choose a new version/directory")
    if any(target.is_relative_to(Path(source).resolve()) or Path(source).resolve().is_relative_to(target) for source in inputs):
        raise OpenSetModelError("Fitted model output must not overlap inputs")
    return target


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    def finite(value):
        if isinstance(value, dict):
            for child in value.values():
                finite(child)
        elif isinstance(value, list):
            for child in value:
                finite(child)
        elif type(value) in (float, int) and not isfinite(value):
            raise ValueError
    finite(data)
    return data


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, allow_nan=False, sort_keys=True, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


@dataclass(frozen=True, eq=False)
class LoadedOpenSetModel:
    parts: FittedOpenSetParts = field(repr=False)
    manifest: dict = field(repr=False)
    threshold: float


def _load_artifacts(root: Path, manifest: dict, frozen: LoadedFrozenConfiguration) -> LoadedOpenSetModel:
    """Hash everything before deserialization; validate fitted state, never predict."""
    config, a = frozen.configuration, frozen.configuration.architecture
    snapshot = json.loads(frozen_configuration_to_json(config))
    if (manifest["schema_version"] != 1 or manifest["status"] != "fitted_and_calibrated" or manifest["completion_state"] != "completed"
            or manifest["frozen_config_sha256"] != frozen.sha256 or manifest["frozen_configuration"] != snapshot
            or manifest["frozen_configuration_id"] != config.configuration_id
            or manifest["frozen_configuration_schema"] != config.schema_version
            or manifest["known_test_payloads_accessed"] is not False or manifest["unknown_holdout_payloads_accessed"] is not False
            or manifest["final_holdouts_evaluated"] is not False or manifest["versions"]["scikit_learn"] != sklearn.__version__
            or set(manifest["artifacts"]) != set(ARTIFACTS)):
        raise ValueError
    for name in ARTIFACTS:
        path = root / name
        if path.is_symlink() or not path.is_file() or _hash(path) != manifest["artifacts"][name]["sha256"]:
            raise ValueError
    threshold = _read_json(root / "threshold.json")
    numeric = threshold["analysis"]["calibration"]["threshold"]
    if (threshold["schema_version"] != 1 or threshold["frozen_config_sha256"] != frozen.sha256
            or threshold["policy"] != snapshot["final_threshold_calibration_policy"]
            or type(numeric) not in (float, int) or not isfinite(numeric) or numeric != manifest["threshold"]
            or threshold["temporal_window_size"] != a.temporal_window_size
            or threshold["novelty_aggregation"] != a.novelty_temporal_aggregation
            or threshold["analysis"] != manifest["calibration_analysis"]):
        raise ValueError
    selector, scaler, svm = (joblib.load(root / name) for name in ARTIFACTS[:3])
    if not isinstance(selector, SelectKBest) or not isinstance(scaler, StandardScaler) or not isinstance(svm, SVC):
        raise ValueError
    for item in (selector, scaler, svm):
        check_is_fitted(item)
    if (selector.k != a.selected_features or selector.score_func is not f_classif or selector.n_features_in_ != a.original_feature_count
            or selector.scores_.shape != (a.original_feature_count,) or selector.get_support().sum() != a.selected_features
            or scaler.n_features_in_ != a.selected_features or svm.n_features_in_ != a.selected_features
            or any(scaler.get_params()[key] != value for key, value in a.scaler_parameters)
            or any(svm.get_params()[key] != value for key, value in a.svm_parameters)
            or svm.decision_function_shape != a.decision_function_shape or svm.break_ties != a.svc_break_ties or svm.fit_status_ != 0):
        raise ValueError
    for vector in (scaler.mean_, scaler.var_, scaler.scale_):
        if vector.shape != (a.selected_features,) or not np.isfinite(vector).all():
            raise ValueError
    if np.any(scaler.scale_ <= 0) or np.any(scaler.var_ < 0):
        raise ValueError
    with np.load(root / "novelty_reference.npz", allow_pickle=False) as values:
        if set(values.files) != {"reference", "reference_labels", "class_ids"}:
            raise ValueError
        reference, labels, ids = values["reference"], values["reference_labels"], values["class_ids"]
    if ids.ndim != 1 or ids.dtype.kind != "U":
        raise ValueError
    class_ids = tuple(str(value) for value in ids)
    for identifier in class_ids:
        validate_identifier(identifier)
    count = len(class_ids)
    if (class_ids != tuple(sorted(set(class_ids))) or count < 2 or count != manifest["identity_count"]
            or reference.shape != (manifest["known_training_samples"], a.selected_features) or reference.dtype != np.float64
            or labels.dtype != np.int32 or labels.shape != (len(reference),)
            or len(reference) < config.final_threshold_calibration_policy.minimum_total_known_reference_samples
            or not np.array_equal(svm.classes_, np.arange(count))
            or svm.shape_fit_ != (len(reference), a.selected_features) or scaler.n_samples_seen_ != len(reference)
            or len(svm.support_) != manifest["support_vector_count"] or not 0 < len(svm.support_) <= len(reference)
            or svm.support_vectors_.shape != (len(svm.support_), a.selected_features)
            or not all(np.isfinite(value).all() for value in (svm.support_vectors_, svm.dual_coef_, svm.intercept_))):
        raise ValueError
    if (manifest["original_feature_count"] != a.original_feature_count or manifest["selected_feature_count"] != a.selected_features
            or manifest["source_representation"] != a.source_representation or manifest["architecture"] != snapshot["architecture"]
            or manifest["known_calibration"]["source_samples"] != len(reference)):
        raise ValueError
    for key in ("known_calibration", "unknown_calibration"):
        counts = manifest[key]
        if (any(type(counts[k]) is not int or counts[k] < 0 for k in ("source_samples", "used_samples", "discarded_samples", "windows"))
                or counts["windows"] < 1 or counts["used_samples"] != counts["windows"] * a.temporal_window_size
                or counts["used_samples"] + counts["discarded_samples"] != counts["source_samples"]):
            raise ValueError
    scorer = KnownDistanceScorer.from_reference_arrays(reference, labels, count, neighbor_counts=(a.novelty_neighbor_count,))
    return LoadedOpenSetModel(FittedOpenSetParts(selector, scaler, svm, scorer, class_ids), manifest, float(numeric))


def model_fingerprints(path: Path) -> tuple[tuple[str, str], ...]:
    """Read-only hashes for every model file, including its manifest."""
    try:
        root = Path(path).resolve()
        names = ("manifest.json", *ARTIFACTS)
        if Path(path).is_symlink() or any((root / name).is_symlink() or not (root / name).is_file() for name in names):
            raise ValueError
        return tuple((name, _hash(root / name)) for name in names)
    except Exception:
        raise OpenSetModelError("Cannot fingerprint complete local model artifacts") from None


def load_open_set_model(path: Path, frozen: LoadedFrozenConfiguration) -> LoadedOpenSetModel:
    """Trusted local files only. No dataset access, predict or transform calls."""
    try:
        root = Path(path).resolve()
        if Path(path).is_symlink() or (root / "manifest.json").is_symlink():
            raise ValueError
        return _load_artifacts(root, _read_json(root / "manifest.json"), frozen)
    except Exception:
        raise OpenSetModelError("Incomplete, incompatible or corrupted fitted open-set model artifacts") from None


def persist_open_set_model(result: FittedOpenSetDevelopmentModel, frozen: LoadedFrozenConfiguration,
                           output: Path, inputs: tuple[Path, ...]) -> None:
    target = validate_model_target(output, inputs)
    stage = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent))
        parts, summary = result.parts, result.summary
        if not isinstance(parts, FittedOpenSetParts):
            raise ValueError
        for name, value in zip(ARTIFACTS[:3], (parts.selector, parts.scaler, parts.svm)):
            joblib.dump(value, stage / name, compress=3)
        reference, labels = parts.reference.reference_arrays()
        np.savez_compressed(stage / "novelty_reference.npz", reference=reference, reference_labels=labels,
                            class_ids=np.asarray(parts.class_ids, dtype=np.str_))
        a = frozen.configuration.architecture
        _write_json(stage / "threshold.json", {"schema_version": 1, "frozen_config_sha256": frozen.sha256,
                    "policy": asdict(frozen.configuration.final_threshold_calibration_policy), "temporal_window_size": a.temporal_window_size,
                    "novelty_aggregation": a.novelty_temporal_aggregation, "analysis": asdict(summary.analysis)})
        for name in ARTIFACTS:
            # Windows _commit/FlushFileBuffers requires a writable handle.
            # These are newly created staging artifacts, never source files.
            with (stage / name).open("r+b") as stream:
                os.fsync(stream.fileno())
        manifest = {**asdict(summary), "schema_version": 1, "status": "fitted_and_calibrated", "completion_state": "completed",
                    "frozen_configuration_id": frozen.configuration.configuration_id, "frozen_configuration_schema": frozen.configuration.schema_version,
                    "frozen_configuration": json.loads(frozen_configuration_to_json(frozen.configuration)),
                    "architecture": json.loads(frozen_configuration_to_json(frozen.configuration))["architecture"],
                    "source_representation": a.source_representation, "threshold": summary.analysis.calibration.threshold,
                    "calibration_analysis": asdict(summary.analysis), "artifacts": {name: {"sha256": _hash(stage / name)} for name in ARTIFACTS},
                    "artifact_privacy": "Biometric-derived local-only; no source sample IDs; pseudonymous class mapping in NPZ only"}
        manifest.pop("analysis")
        manifest = json.loads(json.dumps(manifest, allow_nan=False))
        _load_artifacts(stage, manifest, frozen)  # Validate before the completion marker exists.
        _write_json(stage / "manifest.json", manifest)  # Last file written.
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
        raise OpenSetModelError("Unable to publish fitted model; any retained local staging directory is incomplete and not usable") from None
