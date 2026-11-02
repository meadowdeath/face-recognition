"""Trusted local joblib bundles: integrity validation and atomic no-overwrite publication."""

from dataclasses import asdict, fields
import json
from math import isfinite
import os
from pathlib import Path
import shutil
import tempfile

import joblib

from face_recognition.application.create_v3_cv_protocol import metadata_sha256
from face_recognition.application.v3_deployable_recognition import V3DeployableRecognizer
from face_recognition.domain.models.classification import EvaluationVersions
from face_recognition.domain.models.dataset import CONDITIONS, validate_identifier
from face_recognition.domain.models.rejection import RejectionError
from face_recognition.domain.models.v3_deployable_model import (
    DeployableV3Model, FittedV3Fingerprints, V3CalibrationSummary, V3DeployableAllocation, V3TrainingProvenance,
)
from face_recognition.domain.models.v3_novelty_evaluation import BinaryCalibrationMetrics
from face_recognition.infrastructure.classifiers.sklearn_v3_deployable import (
    SklearnV3DeployableRunner, fitted_v3_fingerprints, validate_fitted_v3_state,
)
from face_recognition.infrastructure.persistence.aggregate_report import validate_aggregate_report_target
from face_recognition.infrastructure.persistence.artifact_integrity import artifact_fingerprints, file_sha256
from face_recognition.infrastructure.persistence.v3_cv_protocol_repository import _object, _read_json
from face_recognition.infrastructure.persistence.v3_mahalanobis_diagnostics_report import _counts, _separation
from face_recognition.infrastructure.persistence.v3_open_set_config import current_implementation_fingerprints, read_frozen_v3_configuration

MODEL_FILE = "model.joblib"
CONFIG_FILE = "frozen_config.json"
MANIFEST_FILE = "manifest.json"
FILES = (MODEL_FILE, CONFIG_FILE, MANIFEST_FILE)


class V3RuntimeCompatibilityError(RejectionError):
    """A validated bundle needs matching runtime/source versions; never bypass it."""


def _check_runtime_versions(expected, actual):
    import re
    differences = []
    for f in fields(expected):
        wanted, installed = getattr(expected, f.name), getattr(actual, f.name)
        if any(not isinstance(v, str) or not re.fullmatch(r"[A-Za-z0-9.+_-]{1,80}", v) for v in (wanted, installed)):
            raise RejectionError("Invalid package version metadata")
        if wanted != installed:
            differences.append(f"{f.name}: bundle={wanted}, runtime={installed}")
    if differences:
        raise V3RuntimeCompatibilityError("V3 runtime mismatch; exact versions are required: " + "; ".join(differences))


def _digest(value):
    import re
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise RejectionError("Invalid model/provenance fingerprint")


def _summary_from_json(raw):
    row = _object(raw, V3CalibrationSummary)
    allocation = _object(row["allocation"], V3DeployableAllocation)
    for name in ("known_identities", "known_fit_sessions", "unknown_calibration_identities", "unknown_calibration_sessions"):
        allocation[name] = tuple(allocation[name])
    row["allocation"] = V3DeployableAllocation(**allocation)
    row["provenance"] = V3TrainingProvenance(**_object(row["provenance"], V3TrainingProvenance))
    row["class_fit_sample_counts"] = tuple(row["class_fit_sample_counts"])
    row["known_calibration"], row["unknown_calibration"] = _counts(row["known_calibration"]), _counts(row["unknown_calibration"])
    metrics = _object(row["calibration_metrics"], BinaryCalibrationMetrics)
    if any(type(v) not in (int, float) or not isfinite(v) or not 0 <= v <= 1 for v in metrics.values()):
        raise RejectionError("Invalid calibration rates")
    row["calibration_metrics"] = BinaryCalibrationMetrics(**metrics)
    row["calibration_diagnostics"] = _separation(row["calibration_diagnostics"], row["known_calibration"].windows, row["unknown_calibration"].windows)
    for name in ("fingerprints_before_calibration", "fingerprints_after_calibration"):
        row[name] = FittedV3Fingerprints(**_object(row[name], FittedV3Fingerprints))
    row["versions"] = EvaluationVersions(**_object(row["versions"], EvaluationVersions))
    return V3CalibrationSummary(**row)


def _validate_model_and_summary(model, summary, config):
    if (type(model) is not DeployableV3Model or type(model.schema_version) is not int or model.schema_version != 1 or model.model_id != "open_set_model_v3"
            or metadata_sha256(model.frozen_configuration) != metadata_sha256(config) or model.class_order != summary.allocation.known_identities
            or type(model.threshold) not in (int, float) or not isfinite(model.threshold) or model.threshold != summary.threshold
            or summary.threshold_transport != "none_same_model_calibration"
            or type(summary.fit_generation_before_calibration) is not int or summary.fit_generation_before_calibration != 1
            or type(summary.fit_generation_after_calibration) is not int or summary.fit_generation_after_calibration != 1
            or summary.fingerprints_before_calibration != summary.fingerprints_after_calibration):
        raise RejectionError("Incomplete or changed deployable state/calibration")
    a = summary.allocation
    for labels in (a.known_identities, a.unknown_calibration_identities, a.known_fit_sessions, a.unknown_calibration_sessions):
        if not isinstance(labels, tuple) or tuple(sorted(set(labels))) != labels:
            raise RejectionError("Invalid ordered population mapping")
        for label in labels: validate_identifier(label)
    validate_identifier(a.known_calibration_session)
    if (len(a.known_identities) < 2 or len(a.unknown_calibration_identities) != 8
            or set(a.known_identities) & set(a.unknown_calibration_identities)
            or len(a.known_fit_sessions) != 3 or a.known_calibration_session <= a.known_fit_sessions[-1]
            or len(a.unknown_calibration_sessions) != 2 or type(a.samples_per_condition) is not int or a.samples_per_condition < 7
            or a.policy != V3DeployableAllocation(a.known_identities, a.known_fit_sessions, a.known_calibration_session,
                a.unknown_calibration_identities, a.unknown_calibration_sessions, a.samples_per_condition).policy):
        raise RejectionError("Invalid chronological allocation")
    n = len(CONDITIONS) * a.samples_per_condition
    expected = (len(a.known_identities) * 3 * n, len(a.known_identities) * n, len(a.unknown_calibration_identities) * 2 * n)
    if (type(summary.known_fit_samples) is not int or any(type(n) is not int for n in summary.class_fit_sample_counts)
            or summary.known_fit_samples != expected[0] or summary.known_calibration.source_samples != expected[1]
            or summary.unknown_calibration.source_samples != expected[2]
            or summary.class_fit_sample_counts != (3 * n,) * len(a.known_identities)
            or model.fitted.identity_count != len(a.known_identities)):
        raise RejectionError("Training/calibration counts or classes differ")
    for counts in (summary.known_calibration, summary.unknown_calibration):
        groups = counts.source_samples // a.samples_per_condition
        if (counts.windows != groups * (a.samples_per_condition // 7) or counts.used_samples != counts.windows * 7
                or counts.discarded_samples != counts.source_samples - counts.used_samples):
            raise RejectionError("Invalid window-7 counts")
    for obj in (summary.fingerprints_before_calibration, summary.fingerprints_after_calibration):
        for f in fields(obj): _digest(getattr(obj, f.name))
    for f in fields(summary.provenance):
        value = getattr(summary.provenance, f.name)
        if f.name.endswith("sha256"): _digest(value)
    filename = summary.provenance.frozen_config_filename
    if not isinstance(filename, str) or Path(filename).name != filename or "/" in filename or "\\" in filename or ":" in filename:
        raise RejectionError("Frozen configuration provenance must contain a filename only")
    for name in ("metrics_scope", "methodology", "limitation", "threshold_transport"):
        if getattr(summary, name) != V3CalibrationSummary.__dataclass_fields__[name].default:
            raise RejectionError("Model construction must retain calibration-only methodological status")
    refs = {r.kind: r for r in config.evidence}
    if (summary.provenance.protocol_file_sha256 != refs["protocol"].sha256
            or summary.provenance.pose_report_file_sha256 != refs["pose_report"].sha256):
        raise RejectionError("Frozen evidence provenance mismatch")
    validate_fitted_v3_state(model.fitted, config, summary.known_fit_samples)
    if (fitted_v3_fingerprints(model.fitted) != summary.fingerprints_after_calibration
            or summary.support_vectors != int(model.fitted.pipeline[-1].n_support_.sum())):
        raise RejectionError("Serialized learned state differs from calibration state")


def _manifest(model, summary, hashes):
    config = model.frozen_configuration
    value = dict(schema_version=1, model_id=model.model_id, status="fitted_and_calibrated", completion_state="completed",
        creation=dict(builder="train_v3_deployable_model", version=1, versions=asdict(summary.versions)),
        representation=config.pipeline.representation, input_features=config.pipeline.input_features,
        selected_features=config.pipeline.selected_features, pipeline=asdict(config.pipeline), temporal=asdict(config.temporal),
        threshold=model.threshold, threshold_objective=config.threshold_policy.objective,
        threshold_tie_breakers=config.threshold_policy.tie_breakers, threshold_transport=config.threshold_policy.threshold_transport,
        class_order=model.class_order, calibration=asdict(summary),
        frozen_configuration=dict(configuration_id=config.configuration_id, configuration_sha256=config.configuration_sha256,
            source_filename=summary.provenance.frozen_config_filename, source_file_sha256=summary.provenance.frozen_config_file_sha256),
        artifacts=tuple(dict(filename=name, sha256=sha) for name, sha in hashes), limitations=config.limitations, manifest_sha256="")
    value["manifest_sha256"] = metadata_sha256(value)
    return value


def _write_bytes(path, content):
    with path.open("xb") as stream:
        stream.write(content); stream.flush(); os.fsync(stream.fileno())


def publish_v3_deployable_bundle(result, frozen_config_path, output_dir, inputs=(), verify_inputs=lambda: None):
    inputs = tuple(inputs) + (Path(frozen_config_path),)
    target = validate_aggregate_report_target(Path(output_dir), inputs)
    config_bytes = Path(frozen_config_path).read_bytes()
    config = read_frozen_v3_configuration(frozen_config_path)
    if file_sha256(Path(frozen_config_path)) != result.summary.provenance.frozen_config_file_sha256:
        raise RejectionError("Frozen config source bytes changed")
    _validate_model_and_summary(result.model, result.summary, config)
    stage = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent))
        _write_bytes(stage / CONFIG_FILE, config_bytes)
        joblib.dump(result.model, stage / MODEL_FILE, compress=3, protocol=5)
        with (stage / MODEL_FILE).open("r+b") as stream: os.fsync(stream.fileno())
        hashes = artifact_fingerprints(stage, (MODEL_FILE, CONFIG_FILE))
        if dict(hashes)[CONFIG_FILE] != result.summary.provenance.frozen_config_file_sha256:
            raise RejectionError("Frozen config copy integrity failure")
        restored = joblib.load(stage / MODEL_FILE)
        _validate_model_and_summary(restored, result.summary, config)
        _validate_model_and_summary(result.model, result.summary, config)
        manifest = _manifest(result.model, result.summary, hashes)
        # Completion manifest is written only after all artifacts round-trip.
        _write_bytes(stage / MANIFEST_FILE, (json.dumps(manifest, sort_keys=True, allow_nan=False, indent=2) + "\n").encode("utf-8"))
        load_v3_deployable_bundle(stage)
        verify_inputs()
        _validate_model_and_summary(result.model, result.summary, config)
        validate_aggregate_report_target(target, inputs)
        os.rename(stage, target)
        stage = None
        return manifest
    except Exception:
        raise RejectionError("Deployable bundle publication failed; no completed bundle published") from None
    finally:
        if stage is not None:
            # Only delete this mkdtemp-created sibling after checking its boundary.
            if (stage.exists() and stage.resolve().parent == target.parent.resolve() and stage.name.startswith(target.name + ".")
                    and stage.name.endswith(".tmp") and not stage.is_symlink()):
                shutil.rmtree(stage)


def load_v3_deployable_bundle(model_dir):
    """Load trusted local artifacts only; hashes provide integrity, not authenticity."""
    try:
        root = Path(model_dir)
        if root.is_symlink() or not root.is_dir() or {p.name for p in root.iterdir()} != set(FILES):
            raise ValueError
        before = artifact_fingerprints(root, FILES)
        manifest = _read_json(root / MANIFEST_FILE)
        if (manifest["schema_version"] != 1 or manifest["status"] != "fitted_and_calibrated"
                or manifest["completion_state"] != "completed" or manifest["model_id"] != "open_set_model_v3"):
            raise ValueError
        checksum = dict(manifest); checksum["manifest_sha256"] = ""
        if manifest["manifest_sha256"] != metadata_sha256(checksum):
            raise ValueError
        hashes = artifact_fingerprints(root, (MODEL_FILE, CONFIG_FILE))
        if manifest["artifacts"] != [dict(filename=n, sha256=s) for n, s in hashes]:
            raise ValueError
        config_metadata = _read_json(root / CONFIG_FILE)
        if config_metadata["implementations"] != [asdict(v) for v in current_implementation_fingerprints()]:
            raise V3RuntimeCompatibilityError("V3 frozen implementation fingerprints differ. Use identical pose-normalizer and Mahalanobis sources, including line endings.")
        config = read_frozen_v3_configuration(root / CONFIG_FILE)
        if dict(hashes)[CONFIG_FILE] != manifest["frozen_configuration"]["source_file_sha256"]:
            raise ValueError
        summary = _summary_from_json(manifest["calibration"])
        runner = SklearnV3DeployableRunner(config)
        _check_runtime_versions(summary.versions, runner.versions)
        model = joblib.load(root / MODEL_FILE)
        _validate_model_and_summary(model, summary, config)
        if metadata_sha256(manifest) != metadata_sha256(_manifest(model, summary, hashes)):
            raise ValueError
        if artifact_fingerprints(root, FILES) != before:
            raise ValueError
        return V3DeployableRecognizer(model, runner, summary.fingerprints_after_calibration)
    except V3RuntimeCompatibilityError:
        raise
    except Exception:
        raise RejectionError("Invalid/incomplete/tampered or incompatible deployable V3 bundle") from None
