"""Immutable development choices and a future calibration contract, never a model."""

from dataclasses import dataclass, field
from math import isfinite
import re

from face_recognition.domain.models.unknown_rejection import RejectionMetrics

FROZEN_CANDIDATE = "mean_5nn__window_7"
DEVELOPMENT_EXPERIMENT = "temporal_distance_novelty_v1"


class ConfigurationFreezeError(ValueError):
    """Sanitized configuration/report validation failure."""


@dataclass(frozen=True)
class OpenSetArchitecture:
    source_representation: str = "all_xyz_v1"
    original_feature_count: int = 1434
    feature_selection_method: str = "SelectKBest"
    feature_selection_score: str = "f_classif"
    selected_features: int = 448
    scaler_type: str = "StandardScaler"
    scaler_parameters: tuple[tuple[str, bool], ...] = (("copy", True), ("with_mean", True), ("with_std", True))
    classifier_type: str = "SVC"
    svm_parameters: tuple[tuple[str, str | float | int | bool], ...] = (
        ("kernel", "rbf"), ("C", 4.0), ("gamma", .001), ("probability", False), ("cache_size", 128), ("random_state", 42),
    )
    decision_function_shape: str = "ovr"
    svc_break_ties: bool = False
    novelty_method: str = "mean_5nn"
    novelty_neighbor_count: int = 5
    novelty_metric: str = "euclidean"
    novelty_distance: str = "Mean of five smallest Euclidean distances to the known reference population; not class-conditioned"
    novelty_score: str = "-novelty_distance; higher means more likely KNOWN"
    temporal_window_size: int = 7
    novelty_temporal_aggregation: str = "median"
    identity_temporal_aggregation: str = "majority_vote"
    identity_tie_breaking: tuple[str, ...] = (
        "largest SVM predicted-identity vote count",
        "highest float64 sum of tied identity's SVM decision score across ALL window frames",
        "first identity in deterministic class order on exact remaining ties",
    )
    class_order: str = "Lexically sorted pseudonymous known IDs; no IDs are stored in this configuration"
    identity_boundary: str = "Neighbors decide KNOWN/UNKNOWN only; accepted identity is aggregated SVM prediction"


@dataclass(frozen=True)
class DevelopmentWindowPolicy:
    order_by: str = "sample_index"
    group_by: tuple[str, ...] = ("person_id", "session_id", "condition")
    anchor: str = "smallest sample_index in each group"
    overlapping: bool = False
    incomplete_windows: str = "discard trailing incomplete remainder; no padding or mixing groups"
    invalid_indices: str = "reject duplicates and gaps"
    future_live_buffer: str = "A rolling latest-7 buffer may be implemented later; not implemented by this freeze"


@dataclass(frozen=True)
class FinalThresholdCalibrationPolicy:
    policy_version: str = "full_development_refit_self_excluded_window_calibration_v1"
    allowed_data_sources: tuple[str, ...] = ("known_train_pool", "unknown_calibration")
    prohibited_data_sources: tuple[str, ...] = ("known_test", "unknown_holdout")
    fitting_partition: str = "known_train_pool"
    reference_partition: str = "known_train_pool"
    known_calibration_partition: str = "known_train_pool"
    unknown_calibration_partition: str = "unknown_calibration"
    ordered_steps: tuple[str, ...] = (
        "Fit ANOVA-448 on ALL known_train_pool features and known labels",
        "Fit StandardScaler on ALL ANOVA-transformed known_train_pool",
        "Fit fixed RBF SVM on ALL selected/scaled known_train_pool",
        "Build mean_5nn reference set from ALL transformed known_train_pool",
        "Score each known calibration query with its own reference row excluded by exact sample key",
        "Score ONLY unknown_calibration queries against the full known reference set",
        "Build frozen non-overlapping 7-sample person/session/condition windows; median novelty, SVM voting",
        "Search deterministic candidate thresholds at WINDOW level using existing signed-score threshold search",
        "Select higher binary balanced accuracy, then lower FAR, lower FRR, higher threshold",
        "Freeze the resulting numeric threshold BEFORE loading either protected holdout payload",
    )
    self_exclusion: bool = True
    self_exclusion_key: tuple[str, ...] = ("person_id", "session_id", "condition", "sample_index")
    self_exclusion_rule: str = "Exclude the query's exact reference row, not all rows with equal feature values; distinct duplicate-valued samples remain eligible"
    minimum_other_reference_samples: int = 5
    minimum_total_known_reference_samples: int = 6
    neighbor_distance_tie_rule: str = "Deterministic canonical sample-key reference order; ties in distances do not change the five-distance mean"
    threshold_search: str = "face_recognition.application.rejection_metrics.calibrate_threshold with allow_negative=True"
    threshold_candidates: str = "Sorted unique observed scores, finite interior midpoints, accept-all boundary min(0,min_score), finite nextafter(max_score,+inf) reject-all boundary"
    selection_rule: tuple[str, ...] = ("higher binary balanced accuracy", "lower FAR", "lower FRR", "higher threshold")
    decision_rule: str = "Accept aggregated SVM identity if median novelty score >= threshold, otherwise UNKNOWN"
    fold_threshold_transfer: str = "Do not copy or average development fold thresholds as the final numeric threshold"
    limitation: str = "Known calibration uses refit training observations with neighbor self-exclusion, not independent out-of-session observations; later holdouts must measure threshold transfer"


@dataclass(frozen=True)
class FrozenMetricDefinitions:
    version: str = "open_set_window_metrics_v1"
    unit: str = "One complete non-overlapping temporal window"
    known_positive: bool = True
    definitions: tuple[tuple[str, str], ...] = (
        ("ROC_AUC", "P(known confidence > unknown confidence) + 0.5*P(tie), computed from continuous confidence"),
        ("EER", "Finite candidate minimizing abs(FAR-FRR); ties lower FAR, lower FRR, higher threshold; report (FAR+FRR)/2, no interpolation"),
        ("FAR", "accepted unknown windows / all unknown windows"),
        ("FRR", "rejected known windows / all known windows"),
        ("known_acceptance", "accepted known windows / all known windows"),
        ("unknown_rejection", "rejected unknown windows / all unknown windows"),
        ("known_correct_identification", "accepted AND correctly identified known windows / all known windows"),
        ("known_misidentification", "accepted BUT incorrectly identified known windows / all known windows"),
        ("open_set_exact_accuracy", "(accepted correctly identified known windows + rejected unknown windows) / all known and unknown windows"),
        ("binary_balanced_accuracy", "(known acceptance rate + unknown rejection rate) / 2"),
        ("binary_macro_F1", "Unweighted mean of binary KNOWN and UNKNOWN F1; zero denominator contributes 0"),
        ("closed_set_accuracy", "Correct SVM identities / all known windows BEFORE novelty rejection"),
        ("closed_set_macro_precision", "Mean per-known-class TP/(TP+FP) before rejection, deterministic class order, zero denominator contributes 0"),
        ("closed_set_macro_recall", "Mean per-known-class TP/(TP+FN) before rejection, deterministic class order, zero denominator contributes 0"),
        ("closed_set_macro_F1", "Mean per-known-class 2*TP/(2*TP+FP+FN) before rejection, deterministic class order, zero denominator contributes 0"),
    )
    development_aggregation: str = "Unweighted two-fold means and population std; unknown windows scored by both models are repeated observations"
    far_operating_points: tuple[float, ...] = (.20, .10, .05)
    far_operating_point_policy: str = "Maximize known acceptance under FAR cap, then lower FAR and higher threshold; acceptance <=1% flagged impractical; diagnostic, not production constraints"


@dataclass(frozen=True)
class DevelopmentReportProvenance:
    sha256: str
    experiment: str = field(default=DEVELOPMENT_EXPERIMENT, init=False)
    schema_version: int = field(default=1, init=False)
    selected_configuration_id: str = field(default=FROZEN_CANDIDATE, init=False)
    completion_state: str = field(default="completed", init=False)
    digest_scope: str = field(default="Exact source aggregate report bytes; no source path stored", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.sha256, str) or re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None:
            raise ConfigurationFreezeError("Invalid aggregate report SHA-256 digest")


@dataclass(frozen=True)
class FrozenDevelopmentEvidence:
    mean_roc_auc: float
    mean_eer: float
    mean_metrics: RejectionMetrics
    known_window_observations: int
    unknown_window_observations: int
    folds: int = field(default=2, init=False)
    scope: str = field(default="Source report's unweighted means at fold-specific balanced-accuracy thresholds; evidence only, not a final numeric threshold", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.mean_metrics, RejectionMetrics):
            raise ConfigurationFreezeError("Invalid aggregate development evidence")
        values = (self.mean_roc_auc, self.mean_eer, *vars(self.mean_metrics).values())
        if any(type(value) not in (float, int) or not isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ConfigurationFreezeError("Development evidence rates must be finite and within zero and one")
        if any(type(value) is not int or value < 1 for value in (self.known_window_observations, self.unknown_window_observations)):
            raise ConfigurationFreezeError("Development evidence counts must be positive integers")


@dataclass(frozen=True)
class FrozenOpenSetConfiguration:
    provenance: DevelopmentReportProvenance
    development_evidence: FrozenDevelopmentEvidence
    schema_version: int = field(default=1, init=False)
    configuration_id: str = field(default="open_set_config_v1", init=False)
    status: str = field(default="development_configuration_frozen", init=False)
    completion_state: str = field(default="frozen", init=False)
    architecture: OpenSetArchitecture = field(default=OpenSetArchitecture(), init=False)
    development_window_policy: DevelopmentWindowPolicy = field(default=DevelopmentWindowPolicy(), init=False)
    final_threshold_calibration_policy: FinalThresholdCalibrationPolicy = field(default=FinalThresholdCalibrationPolicy(), init=False)
    metric_definitions: FrozenMetricDefinitions = field(default=FrozenMetricDefinitions(), init=False)
    final_numeric_threshold: None = field(default=None, init=False)
    final_threshold_status: str = field(default="not_numerically_calibrated; policy_frozen", init=False)
    protected_partitions: tuple[str, ...] = field(default=("known_test", "unknown_holdout"), init=False)
    known_test_payloads_accessed: bool = field(default=False, init=False)
    unknown_holdout_payloads_accessed: bool = field(default=False, init=False)
    final_holdouts_evaluated: bool = field(default=False, init=False)
    final_model_fitted: bool = field(default=False, init=False)
    mutation_policy: str = field(default="No overwrite or unfreeze; different choices require a new version/file and explicit new development experiment", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, DevelopmentReportProvenance) or not isinstance(self.development_evidence, FrozenDevelopmentEvidence):
            raise ConfigurationFreezeError("Frozen configuration requires validated provenance and aggregate evidence")
