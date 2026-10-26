"""Read-only fitted transforms/predictions; no fit, fit_transform or calibration."""

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from threadpoolctl import threadpool_limits

from face_recognition.domain.models.final_evaluation import FinalClosedSet, IdentityClassMetrics
from face_recognition.infrastructure.classifiers.sklearn_baselines import SklearnBaselineRunner, classification_measurements
from face_recognition.infrastructure.classifiers.sklearn_temporal_open_set import FrameInference
from face_recognition.infrastructure.classifiers.sklearn_unknown_rejection import class_decision_scores
from face_recognition.infrastructure.persistence.model_repository import LoadedOpenSetModel


@dataclass(frozen=True, eq=False)
class FinalInference:
    known: FrameInference = field(repr=False)
    unknown: FrameInference = field(repr=False)
    closed_set: FinalClosedSet


class SklearnFinalOpenSetRunner(SklearnBaselineRunner):
    def __init__(self, model: LoadedOpenSetModel) -> None:
        super().__init__()
        self.model = model

    @property
    def frozen_threshold(self) -> float:
        return self.model.threshold

    @property
    def class_ids(self) -> tuple[str, ...]:
        return self.model.parts.class_ids

    def infer(self, known_X: np.ndarray, known_y: np.ndarray, unknown_X: np.ndarray) -> FinalInference:
        parts, identities = self.model.parts, len(self.model.parts.class_ids)
        with threadpool_limits(limits=1):
            def predict(X: np.ndarray) -> FrameInference:
                Z = parts.scaler.transform(parts.selector.transform(X))
                return FrameInference(parts.svm.predict(Z), class_decision_scores(parts.svm, Z, identities), parts.reference.score(Z).confidence)
            known, unknown = predict(known_X), predict(unknown_X)
            metrics, matrix = classification_measurements(known_y, known.predicted, identities)
            precision, recall, f1, support = precision_recall_fscore_support(
                known_y, known.predicted, labels=np.arange(identities), average=None, zero_division=0,
            )
            rows = tuple(IdentityClassMetrics(person, float(p), float(r), float(f), int(n))
                         for person, p, r, f, n in zip(parts.class_ids, precision, recall, f1, support))
        return FinalInference(known, unknown, FinalClosedSet(len(known_y), metrics, parts.class_ids, matrix, rows))
