"""Aggregate diagnostics for the four fixed rejection score definitions."""

from dataclasses import dataclass

from face_recognition.domain.models.anova_svm_tuning import AnovaSvmCandidate
from face_recognition.domain.models.baseline_evaluation import BaselineModelConfiguration, ClassificationMetrics, EvaluationVersions
from face_recognition.domain.models.unknown_rejection import RejectionMetrics, ThresholdCalibration, UnknownRejectionSplit
from face_recognition.domain.models.processed_sample import BASELINE_REPRESENTATION

SCORE_FAMILIES = ("svm_margin", "svm_top1", "centroid_distance", "normalized_centroid_distance")
CENTROID_SCALE_EPSILON = 1e-8
FAR_TARGETS = (.20, .10, .05)


@dataclass(frozen=True)
class ScoreDistribution:
    count: int
    mean: float
    std: float
    median: float
    p10: float
    p25: float
    p75: float
    p90: float


@dataclass(frozen=True)
class EqualErrorPoint:
    eer: float
    threshold: float
    far: float
    frr: float


@dataclass(frozen=True)
class FarOperatingPoint:
    target_far: float
    status: str
    threshold: float | None
    metrics: RejectionMetrics | None
    reason: str | None = None


@dataclass(frozen=True)
class ScoreAnalysis:
    roc_auc: float
    equal_error_point: EqualErrorPoint
    calibration: ThresholdCalibration
    far_operating_points: tuple[FarOperatingPoint, ...]
    known_distribution: ScoreDistribution
    unknown_distribution: ScoreDistribution


@dataclass(frozen=True)
class ScoreSeparation:
    """Threshold-independent diagnostics only; contains no operational calibration."""
    roc_auc: float
    equal_error_point: EqualErrorPoint
    known_distribution: ScoreDistribution
    unknown_distribution: ScoreDistribution
    diagnostic_only: bool = True


@dataclass(frozen=True)
class ScoreFoldResult:
    fold_name: str
    fit_session_ids: tuple[str, ...]
    validation_session_ids: tuple[str, ...]
    fit_samples: int
    known_validation_samples: int
    unknown_calibration_samples: int
    closed_set_metrics: ClassificationMetrics
    analysis: ScoreAnalysis
    at_development_threshold: RejectionMetrics


@dataclass(frozen=True)
class ScoreFoldGaps:
    roc_auc: float
    far: float
    frr: float
    open_set_accuracy: float


@dataclass(frozen=True)
class MarginReferenceComparison:
    roc_auc_delta: float
    eer_delta: float
    far_delta: float
    frr_delta: float
    known_correct_identification_delta: float
    open_set_accuracy_delta: float


@dataclass(frozen=True)
class ScoreEvaluationResult:
    candidate: AnovaSvmCandidate
    score_family: str
    configuration: BaselineModelConfiguration
    folds: tuple[ScoreFoldResult, ...]
    mean_roc_auc: float
    std_roc_auc: float
    mean_eer: float
    std_eer: float
    mean_metrics: RejectionMetrics
    std_metrics: RejectionMetrics
    mean_closed_set_metrics: ClassificationMetrics
    median_development_threshold: float
    absolute_threshold_difference: float
    median_threshold_mean_metrics: RejectionMetrics
    fold_gaps: ScoreFoldGaps
    pooled_known_distribution: ScoreDistribution
    pooled_unknown_distribution: ScoreDistribution
    vs_margin: MarginReferenceComparison | None = None

    @property
    def identifier(self) -> str:
        return f"{self.candidate.identifier}__{self.score_family}"


@dataclass(frozen=True)
class RejectionScoreReport:
    original_feature_count: int
    identity_count: int
    label_order: tuple[str, ...]
    unknown_partition: UnknownRejectionSplit
    results: tuple[ScoreEvaluationResult, ...]
    ranking: tuple[str, ...]
    best_rejection_score_development_candidate: str
    versions: EvaluationVersions
    schema_version: int = 1
    experiment: str = "unknown_rejection_scores_v1"
    source_representation: str = BASELINE_REPRESENTATION
    completion_state: str = "completed"
    score_definitions: tuple[tuple[str, str], ...] = (
        ("svm_margin", "Highest minus second-highest OVR decision score; binary signed d expands to [-d,d]"),
        ("svm_top1", "Maximum class-ordered OVR decision score; binary signed d expands to [-d,d]"),
        ("centroid_distance", "Negative Euclidean distance to SVM-predicted class centroid in selected/scaled space"),
        ("normalized_centroid_distance", "Negative predicted-class centroid distance / fit-only class median distance scale"),
    )
    centroid_scale_epsilon: float = CENTROID_SCALE_EPSILON
    centroid_rule: str = "Class means and median within-class Euclidean distances use known fit rows only; scale=max(median,epsilon)"
    orientation: str = "Larger always means more likely KNOWN; KNOWN is ROC AUC positive class"
    identity_rule: str = "Existing SVM predict behavior; no changes to hyperparameters or vote-tie policy"
    threshold_rule: str = "Observed scores, finite interior midpoints, accept-all/reject-all boundaries; higher BA, lower FAR, lower FRR, higher threshold"
    eer_rule: str = "Finite point minimizing abs(FAR-FRR); ties lower FAR, lower FRR, higher threshold; EER=(FAR+FRR)/2, no interpolation"
    far_point_rule: str = "For FAR<=20%,10%,5% maximize known acceptance, then lower FAR, then higher threshold"
    impractical_rule: str = "Flag known acceptance <=1% as impractical; retain threshold and metrics. Diagnostic convention, not application requirement."
    distribution_rule: str = "Population std; linear percentiles. Pooled distributions combine fold score observations, not unique subjects."
    aggregation_rule: str = "Unweighted fold means/population std at fold-specific BA optima; median-threshold metrics separately"
    ranking_rule: str = "Higher mean AUC, lower mean EER, lower mean FAR, higher correct ID, higher open-set accuracy, smaller threshold gap, fewer features; final tie score name"
    reference_rule: str = "Deltas are candidate score minus same model's svm_margin from these identical fits"
    scope: str = "Development calibration diagnostics only; no independent holdout claim, final model or recognizer persistence"
    transfer_limitation: str = "Score distributions can shift after refitting on more known data; later holdouts must test threshold transfer"
    known_test_payloads_accessed: bool = False
    unknown_holdout_payloads_accessed: bool = False
    native_thread_limit: int = 1
