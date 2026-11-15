"""Deployable V3 state, population policy and calibration-only metadata."""

from dataclasses import dataclass, field
from enum import Enum

from face_recognition.domain.models.classification import EvaluationVersions
from face_recognition.domain.models.score_analysis import ScoreSeparation
from face_recognition.domain.models.temporal import WindowCounts
from face_recognition.domain.models.v3_novelty_evaluation import BinaryCalibrationMetrics
from face_recognition.domain.models.v3_open_set_config import FrozenV3OpenSetConfiguration


class IdentityDecision(str, Enum):
    KNOWN = "KNOWN"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class V3RecognitionDecision:
    decision: IdentityDecision
    identity: str | None
    knownness_score: float
    threshold: float


@dataclass(frozen=True)
class V3FrameObservation:
    """Internal candidate evidence, not an operational recognized identity."""
    candidate_index: int
    candidate_identity: str
    knownness_score: float
    decision_scores: tuple[float, ...] = field(repr=False)


@dataclass(frozen=True)
class V3TemporalOutcome:
    decision: V3RecognitionDecision
    candidate_identity: str  # Debug only when the operational decision is UNKNOWN.


@dataclass(frozen=True)
class V3DeployableAllocation:
    known_identities: tuple[str, ...]
    known_fit_sessions: tuple[str, ...]
    known_calibration_session: str
    unknown_calibration_identities: tuple[str, ...]
    unknown_calibration_sessions: tuple[str, ...]
    samples_per_condition: int
    policy: str = "chronological_first_three_known_sessions_fit_fourth_calibrates_all_development_unknowns_calibrate"


@dataclass(frozen=True)
class FittedV3Fingerprints:
    selected_feature_indices_sha256: str
    feature_selector_sha256: str
    scaler_sha256: str
    identity_svm_sha256: str
    mahalanobis_sha256: str
    overall_sha256: str


@dataclass(frozen=True)
class V3TrainingProvenance:
    frozen_config_filename: str
    frozen_config_file_sha256: str
    protocol_file_sha256: str
    pose_report_file_sha256: str
    pose_catalog_sha256: str
    pose_processing_manifest_sha256: str
    run_processing_manifest_sha256: str
    payload_sha256: str


@dataclass(frozen=True, eq=False)
class DeployableV3Model:
    fitted: object = field(repr=False)
    frozen_configuration: FrozenV3OpenSetConfiguration = field(repr=False)
    class_order: tuple[str, ...]
    threshold: float
    model_id: str = "open_set_model_v3"
    schema_version: int = 1


@dataclass(frozen=True)
class V3CalibrationSummary:
    allocation: V3DeployableAllocation
    provenance: V3TrainingProvenance
    known_fit_samples: int
    class_fit_sample_counts: tuple[int, ...]
    known_calibration: WindowCounts
    unknown_calibration: WindowCounts
    threshold: float
    calibration_metrics: BinaryCalibrationMetrics
    calibration_diagnostics: ScoreSeparation
    fingerprints_before_calibration: FittedV3Fingerprints
    fingerprints_after_calibration: FittedV3Fingerprints
    fit_generation_before_calibration: int
    fit_generation_after_calibration: int
    support_vectors: int
    versions: EvaluationVersions
    threshold_transport: str = "none_same_model_calibration"
    metrics_scope: str = "CALIBRATION METRICS; not independent final test metrics or generalization performance"
    methodology: str = "Deployable construction from frozen V3 development decisions; no new evaluation, selection or cross-validation"
    limitation: str = "No additional untouched V3 holdout exists. V2 remains the last independent external holdout; the cross-validated pose-representation study remains V3 development evidence."


@dataclass(frozen=True, eq=False)
class V3TrainingResult:
    model: DeployableV3Model = field(repr=False)
    summary: V3CalibrationSummary
