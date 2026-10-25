"""Fit known rows only, score known validation and authorized unknown rows."""

from dataclasses import dataclass, field

import numpy as np
from threadpoolctl import threadpool_limits

from face_recognition.application.rejection_metrics import decision_margin
from face_recognition.domain.models.anova_svm_tuning import AnovaSvmCandidate
from face_recognition.domain.models.baseline_evaluation import ClassificationMetrics
from face_recognition.domain.models.feature_reduction import ReductionConfiguration
from face_recognition.infrastructure.classifiers.sklearn_anova_svm import SklearnAnovaSvmRunner
from face_recognition.infrastructure.classifiers.sklearn_baselines import classification_measurements
from face_recognition.infrastructure.classifiers.sklearn_feature_reduction import build_reduction_pipeline


@dataclass(frozen=True, eq=False)
class CalibrationScores:
    known_confidence: np.ndarray = field(repr=False)
    known_correct: np.ndarray = field(repr=False)
    unknown_confidence: np.ndarray = field(repr=False)
    closed_set_metrics: ClassificationMetrics


def _margins(model, X: np.ndarray, identities: int) -> np.ndarray:
    scores = model.decision_function(X)
    if identities == 2 and scores.ndim == 1:
        scores = np.column_stack((-scores, scores))
    if scores.ndim != 2 or scores.shape[1] != identities:
        raise ValueError("SVM score columns do not match the class ordering")
    return decision_margin(scores)


class SklearnUnknownRejectionRunner(SklearnAnovaSvmRunner):
    def score_fold(self, candidate: AnovaSvmCandidate, X_fit: np.ndarray, y_fit: np.ndarray,
                   X_validation: np.ndarray, y_validation: np.ndarray,
                   X_unknown: np.ndarray, identity_count: int) -> CalibrationScores:
        configuration = self.configuration_for(candidate)
        self.validate_inputs(configuration, X_fit, y_fit, X_validation, y_validation)
        if X_unknown.ndim != 2 or X_unknown.shape[1] != X_fit.shape[1] or not len(X_unknown):
            raise ValueError("Unknown calibration feature dimensions are inconsistent")
        model = build_reduction_pipeline(configuration, ReductionConfiguration("anova", k=candidate.k), X_fit.shape[1])
        with threadpool_limits(limits=1):
            model.fit(X_fit, y_fit)
            if (model.named_steps["classifier"].decision_function_shape != "ovr"
                    or not np.array_equal(model.classes_, np.arange(identity_count))):
                raise ValueError("SVM must expose deterministic class-ordered OVR decision scores")
            predictions = model.predict(X_validation)  # Preserve existing closed-set identity behavior.
            metrics, _ = classification_measurements(y_validation, predictions, identity_count)
            known = _margins(model, X_validation, identity_count)
            unknown = _margins(model, X_unknown, identity_count)
        return CalibrationScores(known, predictions == y_validation, unknown, metrics)
