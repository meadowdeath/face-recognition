"""V2 refit/calibration aggregates only; fitted biometric parts stay infrastructure."""

from dataclasses import dataclass

from face_recognition.domain.models.baseline_evaluation import EvaluationVersions
from face_recognition.domain.models.rejection_score_evaluation import ScoreAnalysis, ScoreDistribution
from face_recognition.domain.models.temporal_open_set import WindowCounts
from face_recognition.domain.models.v2_diagnostics import StudyFold


@dataclass(frozen=True)
class V2TransferGuard:
    version: str = "development_score_scale_guard_v1"
    minimum_score_unit: float = .1
    warning_units: float = 1.
    failure_units: float = 3.
    score_unit_definition: str = "max(p90-p10 of pooled known OOF + UNKNOWN fold window scores, 0.1); floor is one tenth of an OVR vote unit"
    decision: str = "Warn above one unit, reject above three units; exact boundary passes. No score alignment or threshold modification."
    scope: str = "Predeclared conservative development sanity guard, not a statistical confidence bound or proof of transfer/generalization"


@dataclass(frozen=True)
class FoldThresholdStability:
    thresholds: tuple[float, ...]
    minimum: float
    maximum: float
    range: float
    mean: float
    median: float
    population_std: float


@dataclass(frozen=True)
class V2CalibrationFold:
    protocol: StudyFold
    fit_samples: int
    known_counts: WindowCounts
    unknown_counts: WindowCounts
    analysis: ScoreAnalysis


@dataclass(frozen=True)
class V2TransferValidation:
    status: str
    guard: V2TransferGuard
    score_unit: float
    fold_threshold_range_units: float
    known_mean_shift: float
    known_median_shift: float
    unknown_mean_shift: float
    unknown_median_shift: float
    maximum_shift_units: float
    warning: bool
    oof_known: ScoreDistribution
    oof_unknown_fold_observations: ScoreDistribution
    final_model_known_development: ScoreDistribution
    final_model_unknown_development: ScoreDistribution
    final_model_scores_usage: str = "DEVELOPMENT transfer diagnostics ONLY; known scores are training resubstitution, never used to select or change the numeric threshold"


@dataclass(frozen=True)
class V2RefitSummary:
    frozen_config_sha256: str
    source_identity_split_sha256: str
    known_training_samples: int
    unknown_unique_source_samples: int
    identity_count: int
    original_feature_count: int
    selected_feature_count: int
    support_vector_count: int
    known_session_ids: tuple[str, ...]
    folds: tuple[V2CalibrationFold, ...]
    known_oof_counts: WindowCounts
    unknown_unique_counts: WindowCounts
    unknown_fold_observation_counts: WindowCounts
    analysis: ScoreAnalysis
    threshold_stability: FoldThresholdStability
    transfer_validation: V2TransferValidation
    versions: EvaluationVersions
    fold_count: int = 3
    calibration_method: str = "cross_session_oof"
    observation_policy: str = "Known windows once OOF; original UNKNOWN windows once per each of three models. Repeated UNKNOWN model observations are not independent physical windows."
    future_known_session_payloads_accessed: bool = False
    future_unknown_payloads_accessed: bool = False
    v2_final_evaluation: str = "not_performed"
    scope: str = "V2 DEVELOPMENT calibration/refit, not final performance. Threshold must not change using future final-test results."
