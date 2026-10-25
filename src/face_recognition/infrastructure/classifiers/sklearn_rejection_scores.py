"""Four scores from one identical fitted identity classifier per fold."""

from dataclasses import dataclass, field

import numpy as np
from threadpoolctl import threadpool_limits

from face_recognition.application.rejection_metrics import decision_margin
from face_recognition.domain.models.anova_svm_tuning import AnovaSvmCandidate
from face_recognition.domain.models.baseline_evaluation import ClassificationMetrics
from face_recognition.infrastructure.classifiers.sklearn_baselines import classification_measurements
from face_recognition.infrastructure.classifiers.sklearn_unknown_rejection import SklearnUnknownRejectionRunner, class_decision_scores
from face_recognition.infrastructure.features.rejection_scores import PredictedClassCentroidScorer, decision_top1


@dataclass(frozen=True, eq=False)
class ScoreObservations:
    family: str
    known: np.ndarray = field(repr=False)
    unknown: np.ndarray = field(repr=False)


@dataclass(frozen=True, eq=False)
class FoldScoreObservations:
    scores: tuple[ScoreObservations, ...] = field(repr=False)
    known_correct: np.ndarray = field(repr=False)
    closed_set_metrics: ClassificationMetrics


class SklearnRejectionScoreRunner(SklearnUnknownRejectionRunner):
    def score_fold_all(self, candidate: AnovaSvmCandidate, X_fit: np.ndarray, y_fit: np.ndarray,
                       X_validation: np.ndarray, y_validation: np.ndarray,
                       X_unknown: np.ndarray, identity_count: int) -> FoldScoreObservations:
        with threadpool_limits(limits=1):
            model = self.fit_pipeline(candidate, X_fit, y_fit, X_validation, y_validation, X_unknown, identity_count)
            transform, classifier = model[:-1], model.named_steps["classifier"]
            Z_fit = transform.transform(X_fit)
            centroids = PredictedClassCentroidScorer().fit(Z_fit, y_fit, identity_count)
            del Z_fit
            Z_known = transform.transform(X_validation)
            predicted = classifier.predict(Z_known)
            metrics, _ = classification_measurements(y_validation, predicted, identity_count)
            known_scores = class_decision_scores(classifier, Z_known, identity_count)
            known_distance, known_normalized = centroids.score_pair(Z_known, predicted)
            del Z_known
            Z_unknown = transform.transform(X_unknown)
            unknown_predicted = classifier.predict(Z_unknown)
            unknown_scores = class_decision_scores(classifier, Z_unknown, identity_count)
            unknown_distance, unknown_normalized = centroids.score_pair(Z_unknown, unknown_predicted)
            observations = (
                ScoreObservations("svm_margin", decision_margin(known_scores), decision_margin(unknown_scores)),
                ScoreObservations("svm_top1", decision_top1(known_scores), decision_top1(unknown_scores)),
                ScoreObservations("centroid_distance", known_distance, unknown_distance),
                ScoreObservations("normalized_centroid_distance", known_normalized, unknown_normalized),
            )
        return FoldScoreObservations(observations, predicted == y_validation, metrics)
