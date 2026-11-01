"""Fixed frozen V2 identity fit and top1 inference; no novelty reference or tuning."""

from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from face_recognition.domain.models.baseline_evaluation import BaselineModelConfiguration
from face_recognition.domain.models.feature_reduction import ReductionConfiguration
from face_recognition.infrastructure.classifiers.sklearn_baselines import SklearnBaselineRunner
from face_recognition.infrastructure.classifiers.sklearn_feature_reduction import build_reduction_pipeline
from face_recognition.infrastructure.classifiers.sklearn_temporal_open_set import FrameInference
from face_recognition.infrastructure.classifiers.sklearn_unknown_rejection import class_decision_scores
from face_recognition.infrastructure.features.rejection_scores import decision_top1


@dataclass(frozen=True, eq=False)
class FittedV2Parts:
    selector: SelectKBest = field(repr=False)
    scaler: StandardScaler = field(repr=False)
    svm: SVC = field(repr=False)
    class_ids: tuple[str, ...] = field(repr=False)


class SklearnV2OpenSetRefitRunner(SklearnBaselineRunner):
    def fit_model(self, config, X, y, class_ids):
        a = config.architecture
        if (X.ndim != 2 or X.shape != (len(y), a.original_feature_count) or not np.isfinite(X).all()
                or not np.array_equal(np.unique(y), np.arange(len(class_ids)))):
            raise ValueError("Invalid frozen V2 known-only training matrix/labels")
        configuration = BaselineModelConfiguration("SVM", a.svm_parameters, a.scaler_parameters)
        with threadpool_limits(limits=1):
            model = build_reduction_pipeline(configuration, ReductionConfiguration("anova", k=a.selected_features), a.original_feature_count)
            model[-1].set_params(decision_function_shape=a.decision_function_shape, break_ties=a.svc_break_ties)
            model.fit(X, y)
            if not np.array_equal(model.classes_, np.arange(len(class_ids))):
                raise ValueError("V2 class order must be deterministic")
        return FittedV2Parts(model[0], model[1], model[2], class_ids)

    def infer(self, parts, X):
        with threadpool_limits(limits=1):
            Z = parts.scaler.transform(parts.selector.transform(X))
            decisions = class_decision_scores(parts.svm, Z, len(parts.class_ids))
            return FrameInference(parts.svm.predict(Z), decisions, (("svm_top1", decision_top1(decisions)),))

    @staticmethod
    def support_count(parts):
        return len(parts.svm.support_)
