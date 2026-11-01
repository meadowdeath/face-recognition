"""Sequential closed-set development evaluation on existing session folds only."""

from typing import Protocol
from statistics import fmean, pstdev

import numpy as np

from face_recognition.application.training_folds import DevelopmentDataset, create_session_folds
from face_recognition.domain.models.baseline_evaluation import (
    BaselineClassifierResult, BaselineEvaluationReport, BaselineFoldResult, BaselineModelConfiguration,
    ClassificationMetrics, EvaluationTiming, EvaluationVersions, FoldMeasurements,
)


class BaselineEvaluationError(ValueError):
    """Sanitized evaluation context without feature values or backend messages."""


class BaselineRunner(Protocol):
    configurations: tuple[BaselineModelConfiguration, ...]
    versions: EvaluationVersions

    def run(self, configuration: BaselineModelConfiguration, X_fit: np.ndarray, y_fit: np.ndarray,
            X_validation: np.ndarray, y_validation: np.ndarray, identity_count: int) -> FoldMeasurements: ...


def summarize_classifier(configuration: BaselineModelConfiguration,
                         folds: tuple[BaselineFoldResult, ...]) -> BaselineClassifierResult:
    """Equal fold weighting preserves the session-level interpretation."""
    def summary(model, field: str, statistic):
        return model(**{name: statistic(getattr(getattr(row.measurements, field), name) for row in folds)
                        for name in model.__dataclass_fields__})

    return BaselineClassifierResult(configuration, folds,
                                    summary(ClassificationMetrics, "metrics", fmean),
                                    summary(ClassificationMetrics, "metrics", pstdev),
                                    summary(EvaluationTiming, "timing", fmean),
                                    summary(EvaluationTiming, "timing", pstdev))


class BaselineIdentityEvaluator:
    def __init__(self, runner: BaselineRunner) -> None:
        self.runner = runner

    def evaluate(self, data: DevelopmentDataset) -> BaselineEvaluationReport:
        context = "Invalid development matrix, labels or partition metadata"
        try:
            # Revalidate without copying arrays: read-only flags are API guards,
            # so callers might otherwise mutate a matrix after initial loading.
            DevelopmentDataset(data.X, data.y, data.samples, data.label_to_person, data.training_sessions)
            if len(data.label_to_person) < 2:
                raise ValueError
            folds = create_session_folds(data)
            results = []
            for configuration in self.runner.configurations:
                rows = []
                for fold in folds:
                    context = f"Baseline evaluation failed: {configuration.name}, {fold.name}"
                    X_fit, y_fit = fold.fit.arrays()
                    X_validation, y_validation = fold.validation.arrays()
                    measurements = self.runner.run(configuration, X_fit, y_fit, X_validation, y_validation,
                                                   len(data.label_to_person))
                    matrix = measurements.confusion_matrix
                    if (len(matrix) != len(data.label_to_person)
                            or any(len(row) != len(data.label_to_person) for row in matrix)
                            or any(type(value) is not int or value < 0 for row in matrix for value in row)
                            or sum(map(sum, matrix)) != len(y_validation)):
                        raise ValueError
                    rows.append(BaselineFoldResult(fold.name, fold.fit_session_ids, fold.validation_session_ids,
                                                   len(y_fit), len(y_validation), len(data.label_to_person),
                                                   data.X.shape[1], data.label_to_person, measurements))
                    # Each backend run discards its estimator before the next;
                    # only aggregate metrics, not predictions, survive here.
                results.append(summarize_classifier(configuration, tuple(rows)))
            if not results:
                raise ValueError
            return BaselineEvaluationReport(data.X.shape[1], len(data.label_to_person), len(data.samples),
                                            data.label_to_person, tuple(results), self.runner.versions)
        except Exception:
            raise BaselineEvaluationError(context) from None
