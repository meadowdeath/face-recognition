"""Identity-level partition metadata and aggregate rejection results only."""

from dataclasses import dataclass

from face_recognition.domain.models.anova_svm_tuning import AnovaSvmCandidate
from face_recognition.domain.models.baseline_evaluation import BaselineModelConfiguration, ClassificationMetrics, EvaluationVersions
from face_recognition.domain.models.dataset import validate_identifier
from face_recognition.domain.models.processed_sample import BASELINE_REPRESENTATION

REJECTION_CANDIDATES = (AnovaSvmCandidate(512, 4.0, .001), AnovaSvmCandidate(448, 4.0, .001))


@dataclass(frozen=True)
class UnknownSplitSpec:
    calibration_ids: tuple[str, ...]
    holdout_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("calibration_ids", "holdout_ids"):
            values = tuple(getattr(self, name))
            if not values or len(set(values)) != len(values):
                raise ValueError("Calibration and holdout identity lists must be nonempty and unique")
            for value in values:
                validate_identifier(value)
            object.__setattr__(self, name, tuple(sorted(values)))
        if set(self.calibration_ids) & set(self.holdout_ids):
            raise ValueError("An unknown identity cannot belong to calibration and holdout")


@dataclass(frozen=True)
class UnknownPartitionCounts:
    identities: int
    sessions: int
    condition_files: int
    samples: int


@dataclass(frozen=True)
class UnknownRejectionSplit:
    specification: UnknownSplitSpec
    source_identity_split_sha256: str
    source_identity_split_schema: int
    source_total_samples: int
    calibration: UnknownPartitionCounts
    holdout: UnknownPartitionCounts
    schema_version: int = 1
    source_representation: str = BASELINE_REPRESENTATION
    completion_state: str = "completed"
    ordering: str = "Identity IDs lexical; all sessions and samples inherit the identity role"
    source_reference: str = "SHA256 of canonical identity-split metadata JSON; no payload digest or absolute source path"


@dataclass(frozen=True)
class RejectionMetrics:
    known_acceptance_rate: float
    known_rejection_rate: float
    unknown_rejection_rate: float
    unknown_false_acceptance_rate: float
    binary_balanced_accuracy: float
    binary_macro_f1: float
    known_correct_identification_rate: float
    known_misidentification_rate: float
    open_set_exact_accuracy: float


@dataclass(frozen=True)
class ThresholdCalibration:
    threshold: float
    metrics: RejectionMetrics
    thresholds_evaluated: int


@dataclass(frozen=True)
class RejectionFoldResult:
    fold_name: str
    fit_session_ids: tuple[str, ...]
    validation_session_ids: tuple[str, ...]
    fit_samples: int
    known_validation_samples: int
    unknown_calibration_samples: int
    closed_set_metrics: ClassificationMetrics
    calibration: ThresholdCalibration
    at_development_threshold: RejectionMetrics


@dataclass(frozen=True)
class RejectionCandidateResult:
    candidate: AnovaSvmCandidate
    configuration: BaselineModelConfiguration
    folds: tuple[RejectionFoldResult, ...]
    mean_metrics: RejectionMetrics
    std_metrics: RejectionMetrics
    median_threshold_mean_metrics: RejectionMetrics
    mean_closed_set_metrics: ClassificationMetrics
    std_closed_set_metrics: ClassificationMetrics
    mean_threshold: float
    median_threshold: float
    absolute_threshold_difference: float
    development_rejection_threshold: float


@dataclass(frozen=True)
class UnknownRejectionReport:
    original_feature_count: int
    identity_count: int
    label_order: tuple[str, ...]
    unknown_partition: UnknownRejectionSplit
    results: tuple[RejectionCandidateResult, ...]
    ranking: tuple[str, ...]
    selected_open_set_development_candidate: AnovaSvmCandidate
    versions: EvaluationVersions
    schema_version: int = 1
    experiment: str = "unknown_rejection_calibration_v1"
    source_representation: str = BASELINE_REPRESENTATION
    completion_state: str = "completed"
    confidence_definition: str = "Largest minus second-largest SVM OVR decision score; binary signed d expands to [-d,d]"
    identity_prediction: str = "Existing SVM predict behavior, in label_order; no change to vote tie breaking"
    acceptance_rule: str = "Accept predicted known identity when margin >= threshold; otherwise UNKNOWN"
    threshold_search: str = "Sorted unique margins, finite interior midpoints, zero and nextafter(max,+inf); no manual grid"
    threshold_selection: str = "Higher binary balanced accuracy, lower unknown FAR, lower known FRR, higher threshold"
    aggregation: str = "Unweighted fold means and population std at fold-optimal thresholds; median-threshold metrics separately"
    ranking_rule: str = (
        "Higher mean fold-optimal open-set exact accuracy, lower unknown FAR, higher known correct identification, "
        "higher binary balanced accuracy, smaller absolute threshold difference, fewer features"
    )
    median_rule: str = "Two-fold median is a + (b-a)/2 after sorting; arithmetic midpoint, not an observed-score quantile"
    calibration_scope: str = "Thresholds selected and described on calibration observations; no independent holdout performance claim"
    transfer_limitation: str = (
        "Margins can shift after refitting on more known development data; future holdout evaluation must measure "
        "transfer of this development threshold. No final recognizer is trained or persisted."
    )
    protected_payloads_accessed: bool = False
    protected_partitions: tuple[str, ...] = ("known_test", "unknown_holdout")
    native_thread_limit: int = 1
