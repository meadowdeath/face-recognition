"""One known-only identity fit per fold; distances never choose the identity."""

from dataclasses import dataclass, field

import numpy as np
from threadpoolctl import threadpool_limits

from face_recognition.application.rejection_metrics import decision_margin
from face_recognition.domain.models.baseline_evaluation import ClassificationMetrics
from face_recognition.domain.models.distance_novelty import NOVELTY_MODEL
from face_recognition.infrastructure.classifiers.sklearn_baselines import classification_measurements
from face_recognition.infrastructure.classifiers.sklearn_rejection_scores import ScoreObservations
from face_recognition.infrastructure.classifiers.sklearn_unknown_rejection import SklearnUnknownRejectionRunner, class_decision_scores
from face_recognition.infrastructure.features.known_distances import KnownDistanceScorer
from face_recognition.infrastructure.features.rejection_scores import PredictedClassCentroidScorer


@dataclass(frozen=True, eq=False)
class DistanceFoldObservations:
    scores: tuple[ScoreObservations, ...] = field(repr=False)
    known_correct: np.ndarray = field(repr=False)
    closed_set_metrics: ClassificationMetrics
    nearest_same_identity_rate: float
    unknown_nearest_majority_concentration: float


class SklearnDistanceNoveltyRunner(SklearnUnknownRejectionRunner):
    def score_novelty_fold(self, X_fit: np.ndarray, y_fit: np.ndarray, X_validation: np.ndarray,
                           y_validation: np.ndarray, X_unknown: np.ndarray, identity_count: int) -> DistanceFoldObservations:
        with threadpool_limits(limits=1):
            model = self.fit_pipeline(NOVELTY_MODEL, X_fit, y_fit, X_validation, y_validation, X_unknown, identity_count)
            transform, classifier = model[:-1], model.named_steps["classifier"]
            Z_fit = transform.transform(X_fit)
            reference = KnownDistanceScorer().fit(Z_fit, y_fit, identity_count)
            centroids = PredictedClassCentroidScorer().fit(Z_fit, y_fit, identity_count)
            del Z_fit

            def infer(X: np.ndarray):
                Z = transform.transform(X)
                predicted = classifier.predict(Z)
                novelty = reference.score(Z)
                margin = decision_margin(class_decision_scores(classifier, Z, identity_count))
                centroid = centroids.score(Z, predicted)
                return predicted, novelty, margin, centroid

            predicted, known, known_margin, known_centroid = infer(X_validation)
            _, unknown, unknown_margin, unknown_centroid = infer(X_unknown)
            closed, _ = classification_measurements(y_validation, predicted, identity_count)
            scores = tuple(ScoreObservations(name, kvalues, uvalues)
                           for (name, kvalues), (_, uvalues) in zip(known.confidence, unknown.confidence))
            scores += (ScoreObservations("svm_margin", known_margin, unknown_margin),
                       ScoreObservations("centroid_distance", known_centroid, unknown_centroid))
            same = float(np.mean(known.nearest_labels == y_validation))
            concentration = float(np.bincount(unknown.nearest_labels, minlength=identity_count).max()) / len(X_unknown)
        return DistanceFoldObservations(scores, predicted == y_validation, closed, same, concentration)
