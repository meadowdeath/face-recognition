"""Aggregate frame-level novelty diagnostics; no reference or sample vectors."""

from dataclasses import dataclass

from face_recognition.domain.models.anova_svm_tuning import AnovaSvmCandidate
from face_recognition.domain.models.baseline_evaluation import BaselineModelConfiguration, ClassificationMetrics, EvaluationVersions
from face_recognition.domain.models.processed_sample import BASELINE_REPRESENTATION
from face_recognition.domain.models.rejection_score_evaluation import ScoreDistribution, ScoreFoldResult
from face_recognition.domain.models.unknown_rejection import RejectionMetrics, UnknownRejectionSplit

NOVELTY_MODEL = AnovaSvmCandidate(448, 4.0, .001)
NOVELTY_NEIGHBORS = (("nearest_1", 1), ("mean_3nn", 3), ("mean_5nn", 5))
REFERENCE_SCORES = ("svm_margin", "centroid_distance")


@dataclass(frozen=True)
class NeighborIdentityDiagnostics:
    fold_name: str
    reference_samples: int
    known_validation_samples: int
    unknown_calibration_samples: int
    nearest_same_identity_rate: float
    unknown_nearest_majority_concentration: float


@dataclass(frozen=True)
class DistanceFoldResult(ScoreFoldResult):
    known_distance_distribution: ScoreDistribution | None
    unknown_distance_distribution: ScoreDistribution | None


@dataclass(frozen=True)
class NoveltyReferenceComparison:
    reference_family: str
    roc_auc_delta: float
    eer_delta: float
    far_delta: float
    frr_delta: float
    known_correct_identification_delta: float
    open_set_accuracy_delta: float


@dataclass(frozen=True)
class DistanceNoveltyResult:
    score_family: str
    neighbor_count: int | None
    folds: tuple[DistanceFoldResult, ...]
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
    known_distance_distribution: ScoreDistribution | None
    unknown_distance_distribution: ScoreDistribution | None
    vs_references: tuple[NoveltyReferenceComparison, ...] = ()


@dataclass(frozen=True)
class DistanceNoveltyReport:
    original_feature_count: int
    identity_count: int
    label_order: tuple[str, ...]
    model: AnovaSvmCandidate
    configuration: BaselineModelConfiguration
    unknown_partition: UnknownRejectionSplit
    novelty_results: tuple[DistanceNoveltyResult, ...]
    reference_results: tuple[DistanceNoveltyResult, ...]
    neighbor_diagnostics: tuple[NeighborIdentityDiagnostics, ...]
    mean_nearest_same_identity_rate: float
    mean_unknown_nearest_majority_concentration: float
    ranking: tuple[str, ...]
    best_distance_based_novelty_development_candidate: str
    versions: EvaluationVersions
    schema_version: int = 1
    experiment: str = "distance_novelty_v1"
    source_representation: str = BASELINE_REPRESENTATION
    completion_state: str = "completed"
    score_definitions: tuple[tuple[str, str], ...] = (
        ("nearest_1", "Negative smallest Euclidean distance to any known fold-fit sample"),
        ("mean_3nn", "Negative mean of three smallest Euclidean distances to known fold-fit samples"),
        ("mean_5nn", "Negative mean of five smallest Euclidean distances to known fold-fit samples"),
        ("svm_margin", "Reference: highest minus second-highest class-ordered SVM OVR decision score"),
        ("centroid_distance", "Reference: negative distance to SVM-predicted class's known-fit centroid"),
    )
    distance_metric: str = "Euclidean in known-fit ANOVA-448 selected, StandardScaler-transformed space"
    distance_implementation: str = "Sequential query batches of at most 128 rows; float64 distance work/reference; no full query-by-reference matrix retained"
    nearest_tie_rule: str = "Exact distance ties choose earliest reference row in deterministic development order; diagnostic only"
    identity_rule: str = "SVM predict supplies identity, never the nearest neighbor; accept when score >= threshold, otherwise UNKNOWN"
    orientation: str = "Higher confidence means KNOWN; novelty score is negative distance; positive distance summaries are separately reported"
    fit_rule: str = "ANOVA, scaler, SVM, neighbor references and centroids use known fold-fit samples only; no validation or unknown fitting"
    threshold_rule: str = "Independent fold/score exact observed/midpoint/boundary search: higher binary BA, lower FAR, lower FRR, higher threshold"
    eer_rule: str = "Closest finite FAR/FRR point; ties lower FAR, lower FRR, higher threshold; EER=(FAR+FRR)/2, no interpolation"
    far_rule: str = "FAR caps 20%,10%,5%; maximize known acceptance, then lower FAR and higher threshold; <=1% acceptance flagged impractical"
    aggregation_rule: str = "Unweighted fold means/population std; median threshold is a+(b-a)/2 for sorted fold thresholds; midpoint metrics separate"
    distribution_rule: str = "Population std and linear percentiles; pooled positive-distance counts are two-fold observations, not additional independent subjects"
    neighbor_diagnostic_rule: str = "Known: nearest reference true label equals validation true label. Unknown: largest aggregate nearest-label frequency / samples; no truth claim or effect on calibration"
    comparison_rule: str = "Novelty minus same-run, same-fitted-model reference; no historical performance or timing constants"
    ranking_rule: str = "Higher mean AUC, lower EER, lower FAR, higher correct known ID, higher open accuracy, lower FRR, smaller neighbor count"
    scope: str = "Frame-level development calibration only; no temporal windows, independent holdout estimate, final model or recognizer persistence"
    transfer_limitation: str = "Reference distances and thresholds can shift after refitting; future protected holdouts must measure transfer"
    known_test_payloads_accessed: bool = False
    unknown_holdout_payloads_accessed: bool = False
    native_thread_limit: int = 1
