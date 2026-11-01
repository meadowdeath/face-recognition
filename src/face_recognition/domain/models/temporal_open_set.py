"""Aggregate offline temporal results, without frame/window biometric sequences."""

from dataclasses import dataclass

from face_recognition.domain.models.anova_svm_tuning import AnovaSvmCandidate
from face_recognition.domain.models.baseline_evaluation import BaselineModelConfiguration, ClassificationMetrics, EvaluationVersions
from face_recognition.domain.models.rejection_score_evaluation import ScoreAnalysis
from face_recognition.domain.models.unknown_rejection import RejectionMetrics, UnknownRejectionSplit
from face_recognition.domain.models.processed_sample import BASELINE_REPRESENTATION

TEMPORAL_MODEL = AnovaSvmCandidate(448, 4.0, .001)
WINDOW_SIZES = (1, 3, 5, 7)
TEMPORAL_SCORES = ("centroid_distance", "svm_margin")


@dataclass(frozen=True)
class WindowCounts:
    source_samples: int
    used_samples: int
    discarded_samples: int
    windows: int


@dataclass(frozen=True)
class VoteRates:
    unanimous_rate: float
    majority_not_unanimous_rate: float
    tie_rate: float


@dataclass(frozen=True)
class VoteStability:
    unanimous_windows: int
    majority_not_unanimous_windows: int
    tied_windows: int
    rates: VoteRates


@dataclass(frozen=True)
class ConditionTemporalDiagnostics:
    condition: str
    known_counts: WindowCounts
    unknown_counts: WindowCounts
    known_correct_identification_rate: float | None
    known_rejection_rate: float | None
    unknown_rejection_rate: float | None


@dataclass(frozen=True)
class TemporalFoldResult:
    fold_name: str
    fit_session_ids: tuple[str, ...]
    validation_session_ids: tuple[str, ...]
    fit_samples: int
    known_counts: WindowCounts
    unknown_counts: WindowCounts
    frame_closed_set_metrics: ClassificationMetrics
    analysis: ScoreAnalysis
    at_development_threshold: RejectionMetrics
    conditions: tuple[ConditionTemporalDiagnostics, ...]
    known_vote_stability: VoteStability


@dataclass(frozen=True)
class FrameReferenceComparison:
    roc_auc_delta: float
    eer_delta: float
    far_delta: float
    frr_delta: float
    known_correct_identification_delta: float
    open_set_accuracy_delta: float


@dataclass(frozen=True)
class TemporalResult:
    score_family: str
    window_size: int
    folds: tuple[TemporalFoldResult, ...]
    mean_roc_auc: float
    std_roc_auc: float
    mean_eer: float
    std_eer: float
    mean_metrics: RejectionMetrics
    std_metrics: RejectionMetrics
    median_development_threshold: float
    absolute_threshold_difference: float
    median_threshold_mean_metrics: RejectionMetrics
    total_known_counts: WindowCounts
    total_unknown_counts: WindowCounts
    mean_vote_rates: VoteRates
    conditions: tuple[ConditionTemporalDiagnostics, ...]
    vs_window1: FrameReferenceComparison | None = None

    @property
    def identifier(self) -> str:
        return f"{self.score_family}__window_{self.window_size}"


@dataclass(frozen=True)
class TemporalOpenSetReport:
    original_feature_count: int
    identity_count: int
    label_order: tuple[str, ...]
    model: AnovaSvmCandidate
    configuration: BaselineModelConfiguration
    unknown_partition: UnknownRejectionSplit
    results: tuple[TemporalResult, ...]
    ranking: tuple[str, ...]
    best_temporal_open_set_development_candidate: str
    versions: EvaluationVersions
    schema_version: int = 1
    experiment: str = "temporal_open_set_v1"
    source_representation: str = BASELINE_REPRESENTATION
    completion_state: str = "completed"
    window_sizes: tuple[int, ...] = WINDOW_SIZES
    score_definitions: tuple[tuple[str, str], ...] = (
        ("centroid_distance", "Negative per-frame Euclidean distance to SVM-predicted known class centroid in selected/scaled space"),
        ("svm_margin", "Per-frame highest minus second-highest OVR decision score; binary signed d expands to [-d,d]"),
    )
    window_policy: str = "Non-overlapping contiguous sample_index chunks within person/session/condition; anchor at smallest index; discard trailing remainder"
    ordering_policy: str = "Sort identities/sessions lexically, conditions in capture order, indices ascending; reject duplicate/gapped indices"
    identity_policy: str = "Largest vote count; ties highest float64 sum of that identity's decision scores across ALL frames; exact remaining ties class order"
    confidence_policy: str = "Median of ALL per-frame scores, including frames not voting for the window winner; larger is more likely KNOWN"
    vote_categories: str = "Unanimous; unique-plurality winner but not unanimous (majority category); tied largest vote count before tie breaking"
    threshold_policy: str = "Independent per fold/score/window size; maximize binary BA, then lower FAR, lower FRR, higher threshold"
    median_rule: str = "Two-fold development threshold is sorted a+(b-a)/2; median-threshold metrics separate from fold-optimal metrics"
    aggregation: str = "Unweighted fold means/population std; count totals sum fold observations; condition means omit unavailable rates"
    unknown_count_policy: str = "The same unknown calibration windows are scored by two fold models; totals are observations, not additional independent samples"
    condition_policy: str = "Condition diagnostics use that fold/configuration's global calibrated threshold, never condition-specific tuning"
    far_policy: str = "Inherited diagnostic caps 20%,10%,5%; <=1% known acceptance flagged impractical; not production requirements"
    ranking_policy: str = "Higher mean AUC, lower EER, lower FAR, higher correct ID, higher open accuracy, lower FRR, smaller window; final tie score name"
    comparison_rule: str = "All deltas = candidate window minus same score's window-size-1 reference"
    scope: str = "Offline development only; ordered time-spaced capture samples approximate temporal observations, not continuous-video latency"
    transfer_limitation: str = "Larger windows are not automatically better; inspect improvements and discarded counts. Future live rolling-window and holdout validation remain required."
    known_test_payloads_accessed: bool = False
    unknown_holdout_payloads_accessed: bool = False
    native_thread_limit: int = 1
