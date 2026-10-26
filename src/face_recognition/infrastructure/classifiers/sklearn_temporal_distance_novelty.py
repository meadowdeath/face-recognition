"""Ephemeral frame predictions and exactly two novelty scores for windowing."""

import numpy as np
from threadpoolctl import threadpool_limits

from face_recognition.domain.models.distance_novelty import NOVELTY_MODEL
from face_recognition.domain.models.temporal_distance_novelty import TEMPORAL_DISTANCE_NEIGHBORS
from face_recognition.infrastructure.classifiers.sklearn_baselines import classification_measurements
from face_recognition.infrastructure.classifiers.sklearn_temporal_open_set import FrameInference, TemporalFoldInference
from face_recognition.infrastructure.classifiers.sklearn_unknown_rejection import SklearnUnknownRejectionRunner, class_decision_scores
from face_recognition.infrastructure.features.known_distances import KnownDistanceScorer


class SklearnTemporalDistanceNoveltyRunner(SklearnUnknownRejectionRunner):
    def infer_fold(self, X_fit: np.ndarray, y_fit: np.ndarray, X_validation: np.ndarray,
                   y_validation: np.ndarray, X_unknown: np.ndarray, identity_count: int) -> TemporalFoldInference:
        with threadpool_limits(limits=1):
            model = self.fit_pipeline(NOVELTY_MODEL, X_fit, y_fit, X_validation, y_validation, X_unknown, identity_count)
            transform, classifier = model[:-1], model.named_steps["classifier"]
            Z_fit = transform.transform(X_fit)
            reference = KnownDistanceScorer(neighbor_counts=tuple(count for _, count in TEMPORAL_DISTANCE_NEIGHBORS)).fit(
                Z_fit, y_fit, identity_count,
            )
            del Z_fit

            def infer(X: np.ndarray) -> FrameInference:
                Z = transform.transform(X)
                predicted = classifier.predict(Z)
                decisions = class_decision_scores(classifier, Z, identity_count)
                return FrameInference(predicted, decisions, reference.score(Z).confidence)

            known, unknown = infer(X_validation), infer(X_unknown)
            metrics, _ = classification_measurements(y_validation, known.predicted, identity_count)
        return TemporalFoldInference(known, unknown, metrics)
