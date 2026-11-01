"""V2 development-only aggregate diagnostics; V1 artifacts are immutable references."""

from dataclasses import dataclass

from face_recognition.domain.models.final_evaluation import FinalClosedSet
from face_recognition.domain.models.rejection_score_evaluation import ScoreDistribution

V2_SCORES = ("svm_top1", "svm_margin", "mean_5nn", "centroid_distance")
V2_STRATEGIES = (("svm_top1_global", "svm_top1", False), ("svm_margin_global", "svm_margin", False),
                 ("mean_5nn_global", "mean_5nn", False), ("centroid_global", "centroid_distance", False),
                 ("per_identity_top1", "svm_top1", True), ("per_identity_mean_5nn", "mean_5nn", True))
MIN_IDENTITY_SUPPORT = 10


@dataclass(frozen=True)
class V1Reference:
    report_sha256: str
    identity_split_sha256: str
    closed_set_accuracy: float
    closed_set_macro_f1: float
    roc_auc: float
    eer: float
    far: float
    frr: float
    known_correct_identification: float
    open_set_accuracy: float


@dataclass(frozen=True)
class StudyFold:
    name: str
    fit_sessions: tuple[str, ...]
    calibration_session: str


@dataclass(frozen=True)
class ConfusionSummary:
    labels: tuple[str, ...]
    observation_counts: tuple[tuple[int, ...], ...]
    row_normalized_rates: tuple[tuple[float, ...], ...]
    role_adjusted_counts: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class DistributionSummary:
    count: int
    statistics: ScoreDistribution | None


@dataclass(frozen=True)
class IdentityThreshold:
    person_id: str
    known_support: int
    unknown_support: int
    eligible_known_support: int
    eligible_unknown_support: int
    threshold: float | None
    used_threshold: float
    available: bool


@dataclass(frozen=True)
class StrategyConfiguration:
    strategy: str
    score_family: str
    per_identity: bool
    window: int
    strict_majority_gate: bool

    @property
    def identifier(self):
        return f"{self.strategy}__window_{self.window}__{'strict_majority' if self.strict_majority_gate else 'normal'}"


@dataclass(frozen=True)
class V2DiagnosticReport:
    v1_baseline: V1Reference
    known_identity_ids: tuple[str, ...]
    unknown_development_identity_count: int
    known_source_samples: int
    unknown_source_samples: int
    folds: tuple[StudyFold, ...]
    closed_set_frame_oof: FinalClosedSet
    score_distributions: tuple[dict, ...]
    candidates: tuple[dict, ...]
    ranking: tuple[str, ...]
    best_v2_development_candidate: str
    schema_version: int = 1
    experiment: str = "v2_open_set_diagnostics_v1"
    status: str = "development_diagnostics_completed"
    source_representation: str = "all_xyz_v1"
    original_feature_count: int = 1434
    identity_model_fit_count: int = 3
    known_data_role: str = "All ORIGINAL known_train_pool + known_test sessions are V2 development, each scored once out of session"
    unknown_data_role: str = "All ORIGINAL unknown_calibration + unknown_holdout identities are V2 development, rotated at identity level"
    identity_model: tuple = (("selector", "SelectKBest(f_classif)"), ("selected_features", 448),
                             ("scaler", "StandardScaler"), ("classifier", "SVC"), ("kernel", "rbf"),
                             ("C", 4.0), ("gamma", .001), ("probability", False))
    temporal_policy: str = "Windows 1/3/5/7, ordered by sample_index within person/session/condition, non-overlapping, discard incomplete tails; median confidence; plurality vote then summed SVM decisions then class order"
    score_definitions: tuple = (("svm_top1", "maximum OVR decision score"), ("svm_margin", "largest minus second largest OVR decision score"),
                                ("mean_5nn", "negative mean five nearest Euclidean fit-reference distances"),
                                ("centroid_distance", "negative Euclidean distance to predicted-class fit centroid"))
    threshold_policy: str = "Known out-of-session plus calibration-only unknown identities; sorted observed scores, interior midpoints, finite boundaries; higher BA, lower FAR, lower FRR, higher threshold; accept score>=threshold and optional strict-majority gate"
    v2_frozen: bool = False
    production_model_persisted: bool = False
    v1_immutable: bool = True
    future_data_used: bool = False
    future_evaluation_required: str = "Freeze V2 before collecting/evaluating a NEW known session (conceptually session_04) and NEW unknown identities; about three preserves V1 comparability, more improves breadth"
    scope: str = "CURRENT V1 data are V2 DEVELOPMENT data, not independent final V2 performance"
    known_calibration_caveat: str = "Known out-of-session data calibrate and describe thresholds; they are not an independent threshold-validation population. Unknown evaluation rotates held-out identities."
    unknown_cv: str = "Leave one existing unknown identity out, lexical identity order; output rotation indices only, never unknown per-person scores"
    support_policy: str = "Per predicted known identity require >=10 gate-eligible KNOWN and >=10 gate-eligible calibration UNKNOWN windows; otherwise unavailable with deterministic global fallback"
    gate_policy: str = "Windows 3/5/7: require winner count >= floor(window/2)+1; calibration uses all observations including mandatory gate rejections, keeping the same BA/FAR/FRR/threshold criterion"
    auc_policy: str = "Unweighted mean per known-session/held-unknown-identity cell. AUC/EER describe raw median scalar score, not the additional identity-conditioned thresholds or consistency gate."
    count_policy: str = "Matrices show integer evaluation-observation counts. Known windows repeat once per unknown rotation; role-adjusted counts divide known rows by that rotation count, preventing duplicated known calibration from inflating open accuracy or binary F1. Unknowns are observed under three known-session models."
    ranking_policy: str = "Higher mean raw-score AUC, lower mean EER, lower FAR, lower FRR, higher known correct ID, higher open accuracy, lower accepted misID, global before per-identity then no gate before gate, smaller window; exact ties only, final strategy name"
