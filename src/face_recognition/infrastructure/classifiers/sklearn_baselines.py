"""Batch-only sklearn baselines; unrelated to live Classifier placeholders.

Every run builds a fresh Pipeline and fits it exclusively on X_fit/y_fit.
Reduction experiments reuse timed pipeline execution. No fitted model survives
a run; baseline pipeline construction and classifier parameters remain fixed.
"""

from collections.abc import Callable
from platform import python_version
from time import perf_counter

import numpy as np
import scipy
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits  # Required by scikit-learn; bounds native BLAS/OpenMP too.

from face_recognition.domain.models.baseline_evaluation import (
    BaselineModelConfiguration, ClassificationMetrics, EvaluationTiming, EvaluationVersions, FoldMeasurements,
)

_SCALER = (("copy", True), ("with_mean", True), ("with_std", True))
BASELINE_CONFIGURATIONS = (
    BaselineModelConfiguration("KNN", (("n_neighbors", 5), ("weights", "uniform"), ("algorithm", "brute"),
                                       ("metric", "euclidean"), ("n_jobs", 1)), _SCALER),
    BaselineModelConfiguration("SVM", (("kernel", "rbf"), ("C", 1.0), ("gamma", "scale"),
                                       ("probability", False), ("cache_size", 128), ("random_state", 42)), _SCALER),
    BaselineModelConfiguration("Random Forest", (("n_estimators", 100), ("criterion", "gini"),
                                                 ("max_depth", None), ("min_samples_split", 2),
                                                 ("min_samples_leaf", 1), ("max_features", "sqrt"),
                                                 ("bootstrap", True), ("class_weight", None),
                                                 ("random_state", 42), ("n_jobs", 1)), None),
)


def build_baseline_pipeline(configuration: BaselineModelConfiguration) -> Pipeline:
    """Fresh fit-only steps; copy=True keeps source matrices unchanged."""
    constructors = {"KNN": KNeighborsClassifier, "SVM": SVC, "Random Forest": RandomForestClassifier}
    classifier = constructors[configuration.name](**dict(configuration.classifier_parameters))
    steps = []
    if configuration.scaler_parameters is not None:
        steps.append(("scaler", StandardScaler(**dict(configuration.scaler_parameters))))
    steps.append(("classifier", classifier))
    return Pipeline(steps, memory=None)


def classification_measurements(y_true: np.ndarray, y_predicted: np.ndarray,
                                identity_count: int) -> tuple[ClassificationMetrics, tuple[tuple[int, ...], ...]]:
    labels = np.arange(identity_count, dtype=np.int32)
    if (y_true.ndim != 1 or y_predicted.shape != y_true.shape or not len(y_true)
            or not np.isin(y_true, labels).all() or not np.isin(y_predicted, labels).all()):
        raise ValueError("Invalid classification label metadata")
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_predicted, labels=labels, average="macro", zero_division=0,
    )
    metrics = ClassificationMetrics(float(accuracy_score(y_true, y_predicted)), float(precision), float(recall), float(f1))
    matrix = confusion_matrix(y_true, y_predicted, labels=labels)
    return metrics, tuple(tuple(int(value) for value in row) for row in matrix)


class SklearnBaselineRunner:
    configurations = BASELINE_CONFIGURATIONS

    def __init__(self, clock: Callable[[], float] = perf_counter) -> None:
        self.clock = clock
        self.versions = EvaluationVersions(python_version(), np.__version__, scipy.__version__, sklearn.__version__)

    @staticmethod
    def validate_inputs(configuration: BaselineModelConfiguration, X_fit: np.ndarray, y_fit: np.ndarray,
                        X_validation: np.ndarray, y_validation: np.ndarray) -> None:
        if (X_fit.ndim != 2 or X_validation.ndim != 2 or not X_fit.shape[1]
                or X_fit.shape[1] != X_validation.shape[1] or len(X_fit) != len(y_fit)
                or len(X_validation) != len(y_validation)):
            raise ValueError("Inconsistent baseline matrix dimensions")
        if configuration.name == "KNN" and len(y_fit) < dict(configuration.classifier_parameters)["n_neighbors"]:
            raise ValueError("KNN baseline requires at least five fit samples")

    def run(self, configuration: BaselineModelConfiguration, X_fit: np.ndarray, y_fit: np.ndarray,
            X_validation: np.ndarray, y_validation: np.ndarray, identity_count: int) -> FoldMeasurements:
        self.validate_inputs(configuration, X_fit, y_fit, X_validation, y_validation)
        model = build_baseline_pipeline(configuration)
        return self.evaluate_pipeline(model, X_fit, y_fit, X_validation, y_validation, identity_count)

    def evaluate_pipeline(self, model: Pipeline, X_fit: np.ndarray, y_fit: np.ndarray,
                          X_validation: np.ndarray, y_validation: np.ndarray, identity_count: int) -> FoldMeasurements:
        """Shared fit/predict timer and scoring; caller supplies a fresh pipeline."""
        with threadpool_limits(limits=1):
            start = self.clock()
            model.fit(X_fit, y_fit)
            fitted = self.clock()
            predictions = model.predict(X_validation)
            predicted = self.clock()
            metrics, matrix = classification_measurements(y_validation, predictions, identity_count)
        timing = EvaluationTiming(fitted - start, predicted - fitted,
                                  (predicted - fitted) * 1000 / len(y_validation))
        return FoldMeasurements(metrics, timing, matrix)
