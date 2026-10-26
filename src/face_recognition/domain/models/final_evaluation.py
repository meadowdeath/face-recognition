"""Aggregate final results. No sample-level scores, predictions or biometric vectors."""

from dataclasses import dataclass

from face_recognition.domain.models.baseline_evaluation import ClassificationMetrics, EvaluationVersions
from face_recognition.domain.models.open_set_config import FrozenOpenSetConfiguration
from face_recognition.domain.models.rejection_score_evaluation import ScoreSeparation
from face_recognition.domain.models.temporal_open_set import VoteStability, WindowCounts
from face_recognition.domain.models.unknown_rejection import RejectionMetrics

NO_RETUNING = "Final holdout results must not be used to modify this model version."
CALIBRATION_WARNING = "Calibration performance is not final generalization performance: known calibration reused the known training pool with neighbor self-exclusion, an easier problem than a held-out known session."


@dataclass(frozen=True)
class IdentityClassMetrics:
    person_id: str  # Pseudonymous technical identity only.
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class FinalClosedSet:
    samples: int
    metrics: ClassificationMetrics
    label_order: tuple[str, ...]
    confusion_matrix: tuple[tuple[int, ...], ...]
    per_class: tuple[IdentityClassMetrics, ...]
    confusion_matrix_axes: str = "Rows=true identity; columns=SVM predicted identity; both follow label_order"
    scope: str = "All known_test frames before novelty rejection, including trailing frames discarded from temporal windows"


@dataclass(frozen=True)
class FinalModelContext:
    configuration: FrozenOpenSetConfiguration
    frozen_config_sha256: str
    model_fingerprints: tuple[tuple[str, str], ...]
    threshold: float
    calibration_roc_auc: float
    calibration_balanced_accuracy: float
    model_versions: EvaluationVersions


@dataclass(frozen=True)
class FinalPartitionCounts:
    identities: int
    sessions: int
    condition_files: int
    samples: int


@dataclass(frozen=True)
class FinalConditionMetrics:
    condition: str
    known_source_samples: int
    unknown_source_samples: int
    known_windows: int
    unknown_windows: int
    known_frame_accuracy: float | None
    known_acceptance_rate: float | None
    known_rejection_rate: float | None
    known_correct_identification_rate: float | None
    unknown_rejection_rate: float | None
    unknown_false_acceptance_rate: float | None


@dataclass(frozen=True)
class ComparisonValues:
    roc_auc: float
    eer: float
    far: float
    frr: float
    known_correct_identification: float
    open_set_accuracy: float


@dataclass(frozen=True)
class FinalEvaluationReport:
    context: FinalModelContext
    source_identity_split_sha256: str
    known_test: FinalPartitionCounts
    unknown_holdout: FinalPartitionCounts
    closed_set_known_frames: FinalClosedSet
    frame_novelty_diagnostics: ScoreSeparation
    known_windows: WindowCounts
    unknown_windows: WindowCounts
    operational_metrics: RejectionMetrics
    temporal_novelty_diagnostics: ScoreSeparation
    conditions: tuple[FinalConditionMetrics, ...]
    known_vote_stability: VoteStability
    development_candidate: ComparisonValues
    final_holdout: ComparisonValues
    delta_final_minus_development: ComparisonValues
    evaluation_versions: EvaluationVersions
    schema_version: int = 1
    experiment: str = "final_open_set_evaluation_v1"
    status: str = "final_holdout_evaluation_completed"
    no_retuning: str = NO_RETUNING
    calibration_warning: str = CALIBRATION_WARNING
    diagnostic_threshold_policy: str = "Frame/temporal EER thresholds are diagnostics only; operational metrics ALWAYS use the persisted model threshold"
    model_artifacts_unchanged: bool = True
    known_test_payloads_accessed: bool = True
    unknown_holdout_payloads_accessed: bool = True
    fitting_performed: bool = False
    threshold_calibration_performed: bool = False
