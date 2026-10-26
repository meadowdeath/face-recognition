"""Aggregate-only temporal nearest-known novelty experiment metadata."""

from dataclasses import dataclass

from face_recognition.domain.models.anova_svm_tuning import AnovaSvmCandidate
from face_recognition.domain.models.baseline_evaluation import BaselineModelConfiguration, EvaluationVersions
from face_recognition.domain.models.processed_sample import BASELINE_REPRESENTATION
from face_recognition.domain.models.temporal_open_set import WINDOW_SIZES, TemporalResult
from face_recognition.domain.models.unknown_rejection import UnknownRejectionSplit

TEMPORAL_DISTANCE_NEIGHBORS = (("nearest_1", 1), ("mean_5nn", 5))
TEMPORAL_DISTANCE_SCORES = tuple(name for name, _ in TEMPORAL_DISTANCE_NEIGHBORS)


@dataclass(frozen=True)
class TemporalDistanceNoveltyReport:
    original_feature_count: int
    identity_count: int
    label_order: tuple[str, ...]
    model: AnovaSvmCandidate
    configuration: BaselineModelConfiguration
    unknown_partition: UnknownRejectionSplit
    results: tuple[TemporalResult, ...]
    ranking: tuple[str, ...]
    best_temporal_distance_novelty_development_candidate: str
    versions: EvaluationVersions
    schema_version: int = 1
    experiment: str = "temporal_distance_novelty_v1"
    source_representation: str = BASELINE_REPRESENTATION
    completion_state: str = "completed"
    window_sizes: tuple[int, ...] = WINDOW_SIZES
    neighbor_counts: tuple[tuple[str, int], ...] = TEMPORAL_DISTANCE_NEIGHBORS
    score_definitions: tuple[tuple[str, str], ...] = (
        ("nearest_1", "Per-frame negative smallest Euclidean distance to any known fold-fit sample"),
        ("mean_5nn", "Per-frame negative mean of five smallest Euclidean distances to known fold-fit samples"),
    )
    distance_space: str = "Euclidean in known-fit ANOVA-448 selected / StandardScaler transformed feature space; references use only known fit rows"
    distance_memory_policy: str = "Reuse batched scorer: at most 128 query rows per distance matrix, one owned float64 reference matrix; sequential two-fold fits"
    window_policy: str = "Non-overlapping contiguous sample_index chunks within person/session/condition; anchor at smallest index; discard trailing remainder; no padding"
    ordering_policy: str = "Sort identities/sessions lexically, conditions in capture order, indices ascending; reject duplicate/gapped indices"
    identity_policy: str = "Largest SVM vote count; ties highest float64 sum of that class's decision scores across ALL frames; exact remaining ties class order; neighbors never select identity"
    confidence_policy: str = "Median of ALL per-frame novelty scores, including frames not voting for the winner; larger is more likely KNOWN"
    vote_categories: str = "Unanimous; unique-plurality winner but not unanimous (majority category); tied largest vote count before tie breaking"
    threshold_policy: str = "Independent per fold/method/window; maximize binary balanced accuracy, then lower FAR, lower FRR, higher threshold; no frame-threshold reuse"
    median_rule: str = "Two-fold development threshold is sorted a+(b-a)/2; midpoint metrics separately reported from fold-optimal metrics"
    eer_policy: str = "Closest finite FAR/FRR point; ties lower FAR, lower FRR, higher threshold; EER=(FAR+FRR)/2 without interpolation"
    far_policy: str = "Caps 20%,10%,5%; maximize known acceptance, then lower FAR, higher threshold; acceptance <=1% flagged impractical, not a production constraint"
    aggregation_policy: str = "Unweighted fold means/population std; counts sum fold observations; condition means omit unavailable rates"
    unknown_count_policy: str = "Unknown calibration windows are scored by both fold models; summed counts are observations, not additional independent samples"
    condition_policy: str = "Aggregate across identities using each fold/configuration's global calibrated threshold; no condition-specific tuning"
    comparison_policy: str = "Deltas = candidate window minus SAME method at window size 1, using this run's measurements"
    efficiency_policy: str = "Report used and discarded samples; for 20 samples, window 5 uses all 20 and window 7 uses 14/discards 6; no ranking bonus for window 5"
    ranking_policy: str = "Higher mean AUC, lower EER, lower FAR, higher correct known ID, higher open accuracy, lower FRR, smaller window, smaller neighbor count"
    scope: str = "Offline development calibration only; time-spaced static-condition samples are not continuous video or a live decision-latency measurement"
    transfer_limitation: str = "Temporal aggregation need not improve performance. Refitting changes reference distances; future protected holdouts must assess transfer. No final recognizer or live rolling buffer."
    known_test_payloads_accessed: bool = False
    unknown_holdout_payloads_accessed: bool = False
    native_thread_limit: int = 1
