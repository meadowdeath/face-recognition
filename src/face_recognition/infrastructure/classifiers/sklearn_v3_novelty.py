"""Fixed identity model and fit-only score references; no thresholds or partitions."""

from dataclasses import dataclass, field

import numpy as np
from threadpoolctl import threadpool_limits

from face_recognition.application.rejection_metrics import decision_margin
from face_recognition.application.v3_novelty_data import FrameScores
from face_recognition.domain.models.classification import ClassifierConfiguration
from face_recognition.domain.models.feature_transforms import ReductionConfiguration
from face_recognition.domain.models.v3_novelty_evaluation import CANDIDATES, V3NoveltyConfiguration
from face_recognition.infrastructure.classifiers.sklearn_classification import SklearnClassifierRunner, class_decision_scores, classification_measurements
from face_recognition.infrastructure.classifiers.sklearn_feature_transforms import build_reduction_pipeline
from face_recognition.infrastructure.features.mahalanobis import PredictedClassMahalanobisScorer
from face_recognition.infrastructure.features.rejection_scores import PredictedClassCentroidScorer, decision_top1


@dataclass(frozen=True, eq=False)
class FittedV3Identity:
    pipeline: object = field(repr=False)
    centroid: object = field(repr=False)
    mahalanobis: object = field(repr=False)
    identity_count: int
    methods: tuple[str, ...]


class SklearnV3NoveltyRunner:
    def __init__(self):
        self.versions = SklearnClassifierRunner().versions

    @staticmethod
    def classification_metrics(y, predicted, identities):
        return classification_measurements(y, predicted, identities)[0]

    def fit(self, population, identity_count, methods=CANDIDATES):
        a = V3NoveltyConfiguration()
        if (population.X.shape != (len(population.y), a.input_features) or not np.isfinite(population.X).all()
                or not np.array_equal(np.unique(population.y), np.arange(identity_count))
                or not methods or not set(methods) <= set(CANDIDATES)):
            raise ValueError("Only the complete authorized KNOWN fit population may fit V3 models")
        config = ClassifierConfiguration("SVM", a.svm_parameters, (("copy", True), ("with_mean", True), ("with_std", True)))
        pipeline = build_reduction_pipeline(config, ReductionConfiguration("anova", k=a.selected_features), a.input_features)
        with threadpool_limits(limits=1):
            pipeline.fit(population.X, population.y)
            Z = pipeline[:-1].transform(population.X)
            if Z.shape[1] != 448 or not np.array_equal(pipeline.classes_, np.arange(identity_count)):
                raise ValueError("V3 fitted dimensions/class order mismatch")
            centroid = PredictedClassCentroidScorer().fit(Z, population.y, identity_count) if "centroid_distance" in methods else None
            mahalanobis = PredictedClassMahalanobisScorer().fit(Z, population.y, identity_count) if "mahalanobis_distance" in methods else None
        return FittedV3Identity(pipeline, centroid, mahalanobis, identity_count, tuple(methods))

    def infer(self, model, population):
        with threadpool_limits(limits=1):
            Z = model.pipeline[:-1].transform(population.X)
            classifier = model.pipeline[-1]
            predicted = classifier.predict(Z)
            decisions = class_decision_scores(classifier, Z, model.identity_count)
            scores = []
            for method in model.methods:
                if method == "svm_top1":
                    values = decision_top1(decisions)
                elif method == "svm_margin":
                    values = decision_margin(decisions)
                elif method == "centroid_distance":
                    values = model.centroid.score(Z, predicted)
                else:
                    values = model.mahalanobis.score(Z, predicted)
                scores.append((method, values))
        return FrameScores(predicted, decisions, tuple(scores))
