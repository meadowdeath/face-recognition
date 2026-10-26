"""Exact aggregate-report provenance and atomic external V2 config publication."""

from dataclasses import asdict
import json
from pathlib import Path

from face_recognition.application.freeze_v2_open_set_config import freeze_v2_configuration
from face_recognition.domain.models.baseline_evaluation import ClassificationMetrics
from face_recognition.domain.models.open_set_config import ConfigurationFreezeError
from face_recognition.domain.models.unknown_rejection import RejectionMetrics
from face_recognition.domain.models.v2_diagnostics import StudyFold
from face_recognition.domain.models.v2_open_set_config import FrozenV2OpenSetConfiguration, V2DevelopmentEvidence, V2ReportProvenance
from face_recognition.infrastructure.persistence.baseline_evaluation_report import write_aggregate_report_json
from face_recognition.infrastructure.persistence.open_set_config_repository import read_aggregate_development_report


def frozen_v2_configuration_to_json(config: FrozenV2OpenSetConfiguration) -> str:
    if config != FrozenV2OpenSetConfiguration(config.provenance, config.development_evidence, config.known_cross_session_folds):
        raise ConfigurationFreezeError("Only the exact frozen V2 policy may be published")
    data = asdict(config)
    data["architecture"]["svm_parameters"] = dict(data["architecture"]["svm_parameters"])
    data["architecture"]["scaler_parameters"] = dict(data["architecture"]["scaler_parameters"])
    data["metric_definitions"]["definitions"] = dict(data["metric_definitions"]["definitions"])
    try:
        return json.dumps(data, allow_nan=False, sort_keys=True, indent=2) + "\n"
    except (TypeError, ValueError):
        raise ConfigurationFreezeError("Invalid frozen V2 metadata") from None


def frozen_v2_configuration_from_json(data: dict) -> FrozenV2OpenSetConfiguration:
    """No policy defaults accepted from disk: compare the entire canonical manifest."""
    try:
        p, e = data["provenance"], data["development_evidence"]
        evidence = V2DevelopmentEvidence(ClassificationMetrics(**e["closed_set_frame_metrics"]), e["mean_roc_auc"], e["mean_eer"],
                                         RejectionMetrics(**e["metrics"]), e["known_source_samples"], e["unknown_source_samples"],
                                         e["known_identity_count"], e["unknown_identity_count"], e["known_unique_windows"],
                                         e["unknown_three_model_window_observations"])
        config = FrozenV2OpenSetConfiguration(V2ReportProvenance(p["sha256"], p["v1_final_report_sha256"], p["original_identity_split_sha256"]),
                                             evidence, tuple(StudyFold(f["name"], tuple(f["fit_sessions"]), f["calibration_session"])
                                                             for f in data["known_cross_session_folds"]))
        if json.dumps(data, sort_keys=True, allow_nan=False) != json.dumps(json.loads(frozen_v2_configuration_to_json(config)), sort_keys=True):
            raise ValueError
        return config
    except Exception:
        raise ConfigurationFreezeError("Invalid, incomplete or modified frozen V2 configuration") from None


def read_frozen_v2_configuration(path: Path) -> FrozenV2OpenSetConfiguration:
    return frozen_v2_configuration_from_json(read_aggregate_development_report(path).payload)


def write_frozen_v2_configuration(config: FrozenV2OpenSetConfiguration, output: Path, diagnostic_report: Path):
    document = read_aggregate_development_report(diagnostic_report)
    if document.sha256 != config.provenance.sha256:
        raise ConfigurationFreezeError("Aggregate diagnostic report changed before V2 publication")
    if config != freeze_v2_configuration(document.payload, document.sha256):
        raise ConfigurationFreezeError("Frozen V2 evidence/policy does not match its aggregate source report")
    write_aggregate_report_json(frozen_v2_configuration_to_json(config), output, (diagnostic_report,))
