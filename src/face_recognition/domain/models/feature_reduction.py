"""Reduction specifications and aggregate-only development experiment results."""

from dataclasses import dataclass
from math import isfinite

from face_recognition.domain.models.baseline_evaluation import BaselineClassifierResult, EvaluationVersions, FoldMeasurements
from face_recognition.domain.models.processed_sample import BASELINE_REPRESENTATION


@dataclass(frozen=True)
class ReductionConfiguration:
    method: str
    k: int | None = None
    explained_variance: float | None = None

    def __post_init__(self) -> None:
        if self.method == "baseline":
            valid = self.k is None and self.explained_variance is None
        elif self.method == "anova":
            valid = type(self.k) is int and self.k > 0 and self.explained_variance is None
        elif self.method == "pca":
            valid = (self.k is None and type(self.explained_variance) in (int, float)
                     and isfinite(self.explained_variance) and 0 < self.explained_variance < 1)
        else:
            valid = False
        if not valid:
            raise ValueError("Invalid feature reduction configuration")

    @property
    def identifier(self) -> str:
        if self.method == "anova":
            return f"anova_k_{self.k}"
        if self.method == "pca":
            return f"pca_{self.explained_variance}"
        return "baseline"


DEFAULT_REDUCTIONS = (
    ReductionConfiguration("baseline"),
    *(ReductionConfiguration("anova", k=k) for k in (256, 512, 768, 1024)),
    *(ReductionConfiguration("pca", explained_variance=value) for value in (.90, .95, .99)),
)


@dataclass(frozen=True)
class FittedReductionMetadata:
    retained_features: int
    selected_feature_indices: tuple[int, ...] | None = None
    actual_explained_variance: float | None = None


@dataclass(frozen=True)
class ReducedMeasurements:
    measurements: FoldMeasurements
    transformation: FittedReductionMetadata


@dataclass(frozen=True)
class FoldReductionMetadata:
    fold_name: str
    transformation: FittedReductionMetadata


@dataclass(frozen=True)
class BaselineComparison:
    macro_f1_delta: float
    accuracy_delta: float
    feature_count_reduction: float
    feature_count_reduction_percent: float
    prediction_latency_delta_ms: float


@dataclass(frozen=True)
class ReductionClassifierResult:
    reduction: ReductionConfiguration
    evaluation: BaselineClassifierResult
    transformations: tuple[FoldReductionMetadata, ...]
    mean_retained_features: float
    std_retained_features: float
    comparison: BaselineComparison


@dataclass(frozen=True)
class RankedReduction:
    rank: int
    reduction_id: str
    classifier_name: str


@dataclass(frozen=True)
class FeatureReductionReport:
    original_feature_count: int
    identity_count: int
    development_samples: int
    label_order: tuple[str, ...]
    results: tuple[ReductionClassifierResult, ...]
    ranking: tuple[RankedReduction, ...]
    versions: EvaluationVersions
    schema_version: int = 1
    experiment: str = "feature_reduction_v1"
    source_representation: str = BASELINE_REPRESENTATION
    completion_state: str = "completed"
    development_partition: str = "known_train_pool"
    protected_partitions: tuple[str, ...] = ("known_test", "unknown_pool")
    protocol: str = "Two opposite whole-session folds; all transformations fitted independently on fit rows only"
    aggregation: str = "Unweighted fold means and population standard deviations; per-fold confusion matrices"
    ranking_rule: str = (
        "Mean macro F1 descending, mean accuracy descending, mean retained dimensions ascending, "
        "mean prediction latency ascending, then reduction ID and classifier name; exploratory, not final selection"
    )
    comparison_rule: str = (
        "Metric and latency deltas = candidate - matching classifier baseline from this run; "
        "feature reduction = original - mean retained dimensions; percentage relative to original dimension"
    )
    timing_scope: str = (
        "perf_counter; fit includes all transformations and classifier fitting; prediction includes all validation "
        "transformations; metrics/metadata excluded; latency is batch prediction time / sample count"
    )
    pca_configuration: str = "StandardScaler(copy=True), PCA(svd_solver=full, copy=True, whiten=False); no second scaler"
    zero_division: int = 0
    native_thread_limit: int = 1
    confusion_matrix_axes: str = "Rows true, columns predicted; both follow label_order"
