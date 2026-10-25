"""Synthetic, varied feature matrices for leakage and numerical tests."""

from dataclasses import replace

import numpy as np

from face_recognition.domain.models.baseline_evaluation import ClassificationMetrics, EvaluationTiming, FoldMeasurements
from face_recognition.domain.models.feature_reduction import FittedReductionMetadata, ReducedMeasurements, ReductionConfiguration
from face_recognition.infrastructure.classifiers.sklearn_baselines import BASELINE_CONFIGURATIONS, SklearnBaselineRunner
from tests.baseline_fakes import development_data

SMALL_REDUCTIONS = (ReductionConfiguration("baseline"), ReductionConfiguration("anova", k=2),
                    ReductionConfiguration("anova", k=4), ReductionConfiguration("pca", explained_variance=.90),
                    ReductionConfiguration("pca", explained_variance=.95), ReductionConfiguration("pca", explained_variance=.99))


def reduction_data(feature_count=9):
    data = development_data(feature_count=feature_count, samples_per_identity=8)
    X = np.random.default_rng(471).normal(size=data.X.shape).astype(np.float32)
    X[:, 0] += data.y * 8
    return replace(data, X=X)


class RecordingReductionRunner:
    """Exercise the entire default grid without fitting large synthetic models."""
    configurations = BASELINE_CONFIGURATIONS

    def __init__(self):
        self.versions = SklearnBaselineRunner().versions
        self.calls = []

    def run_reduced(self, reduction, configuration, X_fit, y_fit, X_validation, y_validation, identity_count):
        self.calls.append((reduction.identifier, configuration.name, len(y_fit), len(y_validation)))
        dimension = X_fit.shape[1] if reduction.method == "baseline" else reduction.k if reduction.method == "anova" else 3
        metadata = FittedReductionMetadata(dimension, tuple(range(dimension)) if reduction.method == "anova" else None,
                                           .999 if reduction.method == "pca" else None)
        matrix = tuple(tuple(int(np.sum(y_validation == label)) if label == predicted else 0
                             for predicted in range(identity_count)) for label in range(identity_count))
        return ReducedMeasurements(FoldMeasurements(ClassificationMetrics(1, 1, 1, 1), EvaluationTiming(2, 1, 1), matrix), metadata)
