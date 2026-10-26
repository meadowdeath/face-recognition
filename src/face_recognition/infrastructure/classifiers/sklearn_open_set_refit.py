"""Fit exactly the authoritative frozen pipeline, then score authorized calibration rows."""

from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from face_recognition.domain.models.open_set_config import FrozenOpenSetConfiguration
from face_recognition.domain.models.baseline_evaluation import BaselineModelConfiguration
from face_recognition.domain.models.feature_reduction import ReductionConfiguration
from face_recognition.infrastructure.classifiers.sklearn_baselines import SklearnBaselineRunner
from face_recognition.infrastructure.classifiers.sklearn_feature_reduction import build_reduction_pipeline
from face_recognition.infrastructure.classifiers.sklearn_temporal_open_set import FrameInference
from face_recognition.infrastructure.classifiers.sklearn_unknown_rejection import class_decision_scores
from face_recognition.infrastructure.features.known_distances import KnownDistanceScorer


@dataclass(frozen=True, eq=False)
class FittedOpenSetParts:
    selector: SelectKBest = field(repr=False)
    scaler: StandardScaler = field(repr=False)
    svm: SVC = field(repr=False)
    reference: KnownDistanceScorer = field(repr=False)
    class_ids: tuple[str, ...] = field(repr=False)


@dataclass(frozen=True, eq=False)
class RefitObservations:
    parts: FittedOpenSetParts = field(repr=False)
    known: FrameInference = field(repr=False)
    unknown: FrameInference = field(repr=False)
    support_vector_count: int


class SklearnOpenSetRefitRunner(SklearnBaselineRunner):
    def fit_and_score(self, config: FrozenOpenSetConfiguration, X: np.ndarray, y: np.ndarray,
                      X_unknown: np.ndarray, class_ids: tuple[str, ...]) -> RefitObservations:
        a = config.architecture
        configuration = BaselineModelConfiguration("SVM", a.svm_parameters, a.scaler_parameters)
        with threadpool_limits(limits=1):
            model = build_reduction_pipeline(configuration, ReductionConfiguration("anova", k=a.selected_features), X.shape[1])
            model.named_steps["classifier"].set_params(decision_function_shape=a.decision_function_shape, break_ties=a.svc_break_ties)
            model.fit(X, y)  # Only ALL known_train_pool reaches fit.
            transform, svm = model[:-1], model.named_steps["classifier"]
            Z = transform.transform(X)
            reference = KnownDistanceScorer(neighbor_counts=(a.novelty_neighbor_count,)).fit(Z, y, len(class_ids))

            def infer(values: np.ndarray, excluded: np.ndarray | None = None) -> FrameInference:
                return FrameInference(svm.predict(values), class_decision_scores(svm, values, len(class_ids)),
                                      reference.score(values, exclude_reference_indices=excluded).confidence)

            # Reference row order is exactly the validated known dataset order.
            known = infer(Z, np.arange(len(Z), dtype=np.intp))
            del Z
            unknown = infer(transform.transform(X_unknown))
            parts = FittedOpenSetParts(model[0], model[1], svm, reference, class_ids)
        return RefitObservations(parts, known, unknown, len(svm.support_))
