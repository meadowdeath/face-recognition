"""Ephemeral per-frame inference for one fixed identity model and two scores."""

from dataclasses import dataclass, field

import numpy as np
from threadpoolctl import threadpool_limits

from face_recognition.application.rejection_metrics import decision_margin
from face_recognition.domain.models.baseline_evaluation import ClassificationMetrics
from face_recognition.domain.models.temporal_open_set import TEMPORAL_MODEL
from face_recognition.infrastructure.classifiers.sklearn_baselines import classification_measurements
from face_recognition.infrastructure.classifiers.sklearn_unknown_rejection import SklearnUnknownRejectionRunner, class_decision_scores
from face_recognition.infrastructure.features.rejection_scores import PredictedClassCentroidScorer


@dataclass(frozen=True, eq=False)
class FrameInference:
    predicted: np.ndarray = field(repr=False)
    decisions: np.ndarray = field(repr=False)
    confidence: tuple[tuple[str, np.ndarray], ...] = field(repr=False)


@dataclass(frozen=True, eq=False)
class TemporalFoldInference:
    known: FrameInference = field(repr=False)
    unknown: FrameInference = field(repr=False)
    closed_set_metrics: ClassificationMetrics


class SklearnTemporalOpenSetRunner(SklearnUnknownRejectionRunner):
    def infer_fold(self, X_fit: np.ndarray, y_fit: np.ndarray, X_validation: np.ndarray,
                   y_validation: np.ndarray, X_unknown: np.ndarray, identity_count: int) -> TemporalFoldInference:
        with threadpool_limits(limits=1):
            model = self.fit_pipeline(TEMPORAL_MODEL, X_fit, y_fit, X_validation, y_validation, X_unknown, identity_count)
            transform, classifier = model[:-1], model.named_steps["classifier"]
            Z_fit = transform.transform(X_fit)
            centroids = PredictedClassCentroidScorer().fit(Z_fit, y_fit, identity_count)
            del Z_fit

            def infer(X: np.ndarray) -> FrameInference:
                Z = transform.transform(X)
                predicted = classifier.predict(Z)
                decisions = class_decision_scores(classifier, Z, identity_count)
                return FrameInference(predicted, decisions,
                                      (("centroid_distance", centroids.score(Z, predicted)),
                                       ("svm_margin", decision_margin(decisions))))

            known, unknown = infer(X_validation), infer(X_unknown)
            metrics, _ = classification_measurements(y_validation, known.predicted, identity_count)
        return TemporalFoldInference(known, unknown, metrics)
