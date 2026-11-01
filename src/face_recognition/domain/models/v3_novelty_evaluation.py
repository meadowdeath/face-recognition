"""Fixed V3 experiment definition and aggregate-only nested evaluation results."""

from dataclasses import dataclass, field

from face_recognition.domain.models.classification import ClassificationMetrics, EvaluationVersions
from face_recognition.domain.models.rejection import RejectionMetrics
from face_recognition.domain.models.score_analysis import ScoreDistribution, ScoreSeparation
from face_recognition.domain.models.temporal import WindowCounts

CANDIDATES = ("svm_top1", "svm_margin", "centroid_distance", "mahalanobis_distance")


@dataclass(frozen=True)
class V3NoveltyConfiguration:
    representation: str = "all_xyz_v1"
    input_features: int = 1434
    selected_features: int = 448
    selector: str = "SelectKBest(f_classif)"
    scaler: str = "StandardScaler(copy=True, with_mean=True, with_std=True)"
    classifier: str = "SVC"
    svm_parameters: tuple = (("kernel", "rbf"), ("C", 4.0), ("gamma", .001), ("probability", False),
                             ("cache_size", 128), ("random_state", 42), ("decision_function_shape", "ovr"), ("break_ties", False))
    candidates: tuple[str, ...] = CANDIDATES
    window_size: int = 7
    confidence_aggregation: str = "median"
    identity_aggregation: str = "Plurality; ties sum corresponding SVM decision scores over all frames; remaining exact ties class order"
    consistency_gate: bool = False
    covariance: str = "Per-class LedoitWolf(assume_centered=False, store_precision=False); Cholesky of covariance plus ridge"
    covariance_ridge_relative: float = 1e-8
    covariance_ridge_rule: str = "ridge=1e-8*max(1,trace(shrunk_covariance)/dimension); no unregularized inverse"
    score_orientation: str = "Higher always means more likely KNOWN"
    threshold_rule: str = "Unique observed scores, finite interior midpoints, accept-all/reject-all boundaries; exact integer BA numerator, lower FAR, lower FRR, higher threshold"
    selection_rule: str = "Higher mean inner-validation BA, lower mean FAR, lower mean FRR, higher mean AUC, then declared candidate order"
    pooling_rule: str = "Inner calibration windows only; deduplicate same physical window/fit model, then median across distinct fit models; one pooled score per physical window"
    transfer_rule: str = "Direct unchanged threshold transfer; development-only distributions/acceptance shifts are diagnostics; no normalization or resubstitution calibration"


@dataclass(frozen=True)
class BinaryCalibrationMetrics:
    known_acceptance_rate: float
    known_rejection_rate: float
    unknown_rejection_rate: float
    unknown_false_acceptance_rate: float
    binary_balanced_accuracy: float
    binary_macro_f1: float


@dataclass(frozen=True)
class InnerNoveltyResult:
    split_name: str
    method: str
    threshold: float
    known_calibration_windows: int
    unknown_calibration_windows: int
    known_validation_counts: WindowCounts
    unknown_validation_counts: WindowCounts
    validation_metrics: RejectionMetrics
    validation_diagnostics: ScoreSeparation


@dataclass(frozen=True)
class CandidateSummary:
    method: str
    splits: int
    mean_metrics: RejectionMetrics
    std_metrics: RejectionMetrics
    mean_roc_auc: float
    std_roc_auc: float
    mean_eer: float
    std_eer: float


@dataclass(frozen=True)
class PoolCounts:
    raw_window_observations: int
    distinct_model_window_observations: int
    unique_physical_windows: int
    minimum_models_per_window: int
    maximum_models_per_window: int


@dataclass(frozen=True)
class OuterCalibration:
    threshold: float
    binary_metrics: BinaryCalibrationMetrics
    known_pool: PoolCounts
    unknown_pool: PoolCounts
    diagnostics: ScoreSeparation


@dataclass(frozen=True)
class TransferPopulationDiagnostics:
    cross_fitted: ScoreDistribution
    final_model_development: ScoreDistribution
    mean_shift: float
    median_shift: float
    acceptance_rate_shift: float
    median_outside_cross_fitted_p10_p90: bool


@dataclass(frozen=True)
class ThresholdTransport:
    known: TransferPopulationDiagnostics
    unknown: TransferPopulationDiagnostics
    warning: bool
    policy: str = "Direct transfer, unchanged threshold. Warnings are descriptive, never threshold adjustments."
    known_scope: str = "Final-model KNOWN development scores are training resubstitution, not generalization or threshold calibration"


@dataclass(frozen=True)
class OuterNoveltyResult:
    fold_name: str
    selected_method: str
    threshold: float
    fit_known_samples: int
    known_test_counts: WindowCounts
    unknown_test_counts: WindowCounts
    closed_set_known_frames: ClassificationMetrics
    operational_metrics: RejectionMetrics
    diagnostics: ScoreSeparation
    inner_results: tuple[InnerNoveltyResult, ...]
    inner_summaries: tuple[CandidateSummary, ...]
    inner_ranking: tuple[str, ...]
    calibration: OuterCalibration
    transport: ThresholdTransport
    cached_inner_fit_populations: int
    cache_reuses: int


@dataclass(frozen=True)
class V3NoveltyReport:
    protocol_sha256: str
    protocol_file_sha256: str
    catalog_sha256: str
    processing_manifest_sha256: str
    payload_sha256: str
    configuration_sha256: str
    identity_count: int
    source_samples: int
    outer_results: tuple[OuterNoveltyResult, ...]
    mean_closed_set: ClassificationMetrics
    std_closed_set: ClassificationMetrics
    mean_metrics: RejectionMetrics
    std_metrics: RejectionMetrics
    mean_roc_auc: float
    std_roc_auc: float
    mean_eer: float
    std_eer: float
    selection_frequency: tuple[tuple[str, int], ...]
    inner_comparison: tuple[CandidateSummary, ...]
    versions: EvaluationVersions
    configuration: V3NoveltyConfiguration = field(default_factory=V3NoveltyConfiguration)
    schema_version: int = 1
    experiment: str = "v3_nested_novelty_comparison_v1"
    completion_state: str = "completed"
    interpretation: str = "Outer fold mean/population std estimate nested-CV generalization; no independent untouched final holdout in V3"
    inner_scope: str = "INNER validation comparison is development-only; repeats across splits/folds are not independent physical samples"
    aggregation: str = "Unweighted fold means/population std, including fold-level AUC/EER; never pool raw scores from different outer models"
