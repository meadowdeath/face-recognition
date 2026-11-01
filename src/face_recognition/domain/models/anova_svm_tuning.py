"""Fixed development search and aggregate diagnostics, without biometric values."""

from dataclasses import dataclass

from face_recognition.domain.models.baseline_evaluation import BaselineClassifierResult, EvaluationVersions, FoldMeasurements
from face_recognition.domain.models.feature_reduction import BaselineComparison
from face_recognition.domain.models.processed_sample import BASELINE_REPRESENTATION

ANOVA_K_VALUES = (384, 448, 512, 640, 768)
SVM_C_VALUES = (.5, 1.0, 2.0, 4.0)
SVM_GAMMA_VALUES = ("scale", .001, .01)


@dataclass(frozen=True)
class AnovaSvmCandidate:
    k: int
    C: float
    gamma: str | float
    kernel: str = "rbf"
    probability: bool = False

    def __post_init__(self) -> None:
        if (type(self.k) is not int or self.k not in ANOVA_K_VALUES
                or type(self.C) not in (int, float) or self.C not in SVM_C_VALUES
                or type(self.gamma) not in (str, int, float) or self.gamma not in SVM_GAMMA_VALUES
                or self.kernel != "rbf" or self.probability is not False):
            raise ValueError("Configuration is outside the fixed ANOVA-SVM search space")

    @property
    def identifier(self) -> str:
        return f"anova_k_{self.k}_C_{self.C:g}_gamma_{self.gamma}"


def tuning_candidates() -> tuple[AnovaSvmCandidate, ...]:
    return tuple(AnovaSvmCandidate(k, C, gamma) for k in ANOVA_K_VALUES for C in SVM_C_VALUES for gamma in SVM_GAMMA_VALUES)


REFERENCE_CANDIDATE = AnovaSvmCandidate(512, 1.0, "scale")


@dataclass(frozen=True)
class AnovaSvmSearchSpace:
    k_values: tuple[int, ...] = ANOVA_K_VALUES
    C_values: tuple[float, ...] = SVM_C_VALUES
    gamma_values: tuple[str | float, ...] = SVM_GAMMA_VALUES
    kernel: str = "rbf"
    probability: bool = False


@dataclass(frozen=True)
class SupportVectorCounts:
    total: int
    per_class: tuple[int, ...]

    def __post_init__(self) -> None:
        if (type(self.total) is not int or self.total < 1 or not self.per_class
                or any(type(count) is not int or count < 1 for count in self.per_class)
                or sum(self.per_class) != self.total):
            raise ValueError("Invalid support-vector count metadata")


@dataclass(frozen=True)
class TuningFoldMeasurements:
    measurements: FoldMeasurements
    selected_feature_indices: tuple[int, ...]
    support_vectors: SupportVectorCounts


@dataclass(frozen=True)
class FoldSupportVectors:
    fold_name: str
    counts: SupportVectorCounts


@dataclass(frozen=True)
class SelectionStability:
    k: int
    fold_a_name: str
    fold_b_name: str
    fold_a_indices: tuple[int, ...]
    fold_b_indices: tuple[int, ...]
    shared_features: int
    overlap_percent: float
    jaccard_similarity: float


@dataclass(frozen=True)
class AnovaSvmResult:
    candidate: AnovaSvmCandidate
    evaluation: BaselineClassifierResult
    support_vectors: tuple[FoldSupportVectors, ...]
    worst_fold_macro_f1: float
    fold_f1_gap: float
    vs_full_svm_baseline: BaselineComparison | None = None
    vs_anova512_reference: BaselineComparison | None = None


@dataclass(frozen=True)
class RankedTuningCandidate:
    rank: int
    candidate_id: str


@dataclass(frozen=True)
class AnovaSvmTuningReport:
    original_feature_count: int
    identity_count: int
    development_samples: int
    label_order: tuple[str, ...]
    full_feature_svm_reference: BaselineClassifierResult
    results: tuple[AnovaSvmResult, ...]
    selection_stability: tuple[SelectionStability, ...]
    ranking: tuple[RankedTuningCandidate, ...]
    selected_development_candidate: AnovaSvmCandidate
    versions: EvaluationVersions
    reference_candidate: AnovaSvmCandidate = REFERENCE_CANDIDATE
    search_space: AnovaSvmSearchSpace = AnovaSvmSearchSpace()
    schema_version: int = 1
    experiment: str = "anova_svm_tuning_v1"
    source_representation: str = BASELINE_REPRESENTATION
    completion_state: str = "completed"
    configurations_evaluated: int = 60
    folds_per_configuration: int = 2
    development_partition: str = "known_train_pool"
    protected_partitions: tuple[str, ...] = ("known_test", "unknown_pool")
    protocol: str = "Two opposite whole-session development folds; selector, scaler and SVM fitted on fit rows only"
    aggregation: str = "Unweighted fold means and population standard deviations; per-fold confusion matrices"
    ranking_rule: str = (
        "Mean macro F1 descending, worst-fold macro F1 descending, F1 gap ascending, mean accuracy descending, "
        "k ascending, mean prediction latency ascending; final ties by C then configured gamma order"
    )
    comparison_rule: str = (
        "Candidate minus reference for metric/latency deltas; reference dimension minus k for feature reduction; "
        "percent relative to reference dimension. Full SVM is rerun on these same development folds."
    )
    support_vector_order: str = "per_class counts follow label_order; no support-vector values retained"
    selection_rule: str = "Fit indices must be identical for a given k/fold across C and gamma; never globally selected"
    selection_status: str = "Development candidate only; no final model, unknown calibration or holdout evaluation"
    timing_scope: str = "perf_counter; fit/predict include transformations; prediction latency is a batch mean"
    native_thread_limit: int = 1
    zero_division: int = 0
    confusion_matrix_axes: str = "Rows true, columns predicted; both follow label_order"
