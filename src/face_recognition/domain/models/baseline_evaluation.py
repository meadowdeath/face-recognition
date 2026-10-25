"""Aggregate closed-set development results; never store samples or vectors."""

from dataclasses import dataclass
from math import isfinite

from face_recognition.domain.models.processed_sample import BASELINE_REPRESENTATION

ParameterValue = str | int | float | bool | None


@dataclass(frozen=True)
class BaselineModelConfiguration:
    name: str
    classifier_parameters: tuple[tuple[str, ParameterValue], ...]
    scaler_parameters: tuple[tuple[str, ParameterValue], ...] | None


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float

    def __post_init__(self) -> None:
        if any(not isfinite(value) or not 0 <= value <= 1 for value in (
            self.accuracy, self.macro_precision, self.macro_recall, self.macro_f1,
        )):
            raise ValueError("Classification metrics must be finite and between zero and one")


@dataclass(frozen=True)
class EvaluationTiming:
    fit_seconds: float
    prediction_seconds: float
    mean_prediction_latency_ms: float

    def __post_init__(self) -> None:
        if any(not isfinite(value) or value < 0 for value in (
            self.fit_seconds, self.prediction_seconds, self.mean_prediction_latency_ms,
        )):
            raise ValueError("Evaluation timings must be finite and nonnegative")


@dataclass(frozen=True)
class FoldMeasurements:
    metrics: ClassificationMetrics
    timing: EvaluationTiming
    confusion_matrix: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class BaselineFoldResult:
    fold_name: str
    fit_session_ids: tuple[str, ...]
    validation_session_ids: tuple[str, ...]
    fit_samples: int
    validation_samples: int
    identity_count: int
    feature_count: int
    label_order: tuple[str, ...]
    measurements: FoldMeasurements


@dataclass(frozen=True)
class BaselineClassifierResult:
    configuration: BaselineModelConfiguration
    folds: tuple[BaselineFoldResult, ...]
    mean_metrics: ClassificationMetrics
    std_metrics: ClassificationMetrics
    mean_timing: EvaluationTiming
    std_timing: EvaluationTiming


@dataclass(frozen=True)
class EvaluationVersions:
    python: str
    numpy: str
    scipy: str
    scikit_learn: str


@dataclass(frozen=True)
class BaselineEvaluationReport:
    feature_count: int
    identity_count: int
    development_samples: int
    label_order: tuple[str, ...]
    classifiers: tuple[BaselineClassifierResult, ...]
    versions: EvaluationVersions
    schema_version: int = 1
    experiment: str = "baseline_identity_v1"
    representation: str = BASELINE_REPRESENTATION
    completion_state: str = "completed"
    development_partition: str = "known_train_pool"
    protected_partitions: tuple[str, ...] = ("known_test", "unknown_pool")
    protocol: str = "Two opposite whole-session development folds; no sample-level random split"
    feature_policy: str = "All stored XYZ features in source order; no feature reduction or selection"
    aggregation: str = "Unweighted fold means and population standard deviations; confusion matrices remain per fold"
    zero_division: int = 0
    confusion_matrix_axes: str = "Rows: true identity; columns: predicted identity; both follow label_order"
    timing_scope: str = (
        "perf_counter seconds; fit includes fit-side scaler and classifier; prediction includes validation transform; "
        "metrics and model construction excluded; latency is batch prediction seconds * 1000 / validation samples"
    )
    native_thread_limit: int = 1
