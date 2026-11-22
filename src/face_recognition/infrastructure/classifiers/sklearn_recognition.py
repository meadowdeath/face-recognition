"""Frozen-config-driven fitting and structural validation; same-model inference reused."""

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from sklearn.pipeline import Pipeline
from platform import python_version
import scipy
import sklearn

from face_recognition.application.training_data import FrameScores

from face_recognition.application.recognition_config import validate_recognition_configuration
from face_recognition.domain.models.classification import EvaluationVersions
from face_recognition.domain.models.rejection import RejectionError
from face_recognition.domain.models.v3_deployable_model import FittedV3Fingerprints
from face_recognition.infrastructure.classifiers.sklearn_v3_novelty import FittedV3Identity
from face_recognition.infrastructure.classifiers.model_integrity import fitted_value_sha256
from face_recognition.infrastructure.features.mahalanobis import PredictedClassMahalanobisScorer


def fitted_model_fingerprints(model):
    digest = fitted_value_sha256
    selector = model.pipeline.named_steps["anova"]
    return FittedV3Fingerprints(digest(selector.get_support(indices=True)), digest(selector),
        digest(model.pipeline.named_steps["scaler"]), digest(model.pipeline[-1]), digest(model.mahalanobis), digest(model))


def validate_fitted_model_state(model, config, known_fit_samples=None):
    """Inspect fitted shapes/parameters, never perform inference or fitting."""
    validate_recognition_configuration(config)
    p = config.pipeline
    if not isinstance(model, FittedV3Identity) or model.methods != (p.novelty_method,) or model.centroid is not None:
        raise RejectionError("Invalid V3 fitted components")
    if tuple(model.pipeline.named_steps) != ("anova", "scaler", "classifier"):
        raise RejectionError("Invalid V3 fitted pipeline order")
    selector, scaler, svm = model.pipeline.named_steps["anova"], model.pipeline.named_steps["scaler"], model.pipeline[-1]
    if (type(selector) is not SelectKBest or selector.score_func is not f_classif or selector.k != p.selected_features
            or selector.n_features_in_ != p.input_features or len(selector.get_support(indices=True)) != p.selected_features
            or type(scaler) is not StandardScaler or scaler.n_features_in_ != p.selected_features
            or scaler.get_params() != dict(copy=True, with_mean=True, with_std=True)
            or type(svm) is not SVC or svm.n_features_in_ != p.selected_features
            or any(svm.get_params()[key] != value for key, value in p.svm_parameters)
            or type(model.identity_count) is not int or model.identity_count < 2
            or not np.array_equal(svm.classes_, np.arange(model.identity_count))
            or svm.support_vectors_.ndim != 2 or svm.support_vectors_.shape[1] != p.selected_features
            or not np.isfinite(svm.support_vectors_).all()):
        raise RejectionError("Fitted V3 estimator configuration/dimensions differ from frozen configuration")
    for value in (scaler.mean_, scaler.var_, scaler.scale_):
        if np.shape(value) != (p.selected_features,) or not np.isfinite(value).all():
            raise RejectionError("Invalid scaler learned state")
    if np.any(scaler.var_ < 0) or np.any(scaler.scale_ <= 0):
        raise RejectionError("Invalid scaler statistics")
    mah = model.mahalanobis
    if (type(mah) is not PredictedClassMahalanobisScorer
            or np.shape(mah._centers) != (model.identity_count, p.selected_features)
            or not np.isfinite(mah._centers).all() or len(mah._factors) != model.identity_count
            or (known_fit_samples is not None and mah.fit_samples != known_fit_samples)):
        raise RejectionError("Invalid Mahalanobis reference dimensions/counts")
    for factor in mah._factors:
        if (factor.shape != (p.selected_features, p.selected_features) or not np.isfinite(factor).all()
                or np.any(np.diag(factor) <= 0) or np.any(np.triu(factor, k=1) != 0)):
            raise RejectionError("Invalid Mahalanobis Cholesky state")


class RecognitionModelRunner:
    def __init__(self, frozen_config):
        self.fit_count = 0
        self.versions = EvaluationVersions(python_version(), np.__version__, scipy.__version__, sklearn.__version__)
        validate_recognition_configuration(frozen_config)
        self.config = frozen_config

    def fit(self, population, identity_count, methods=("mahalanobis_distance",)):
        if self.fit_count != 0 or methods != (self.config.pipeline.novelty_method,):
            raise RejectionError("Deployable runner permits exactly one fit of the frozen method")
        p = self.config.pipeline
        if (population.X.shape != (len(population.y), p.input_features) or not np.isfinite(population.X).all()
                or not np.array_equal(np.unique(population.y), np.arange(identity_count))):
            raise RejectionError("Only complete KNOWN fitting data may fit the deployable model")
        self.fit_count += 1
        pipeline = Pipeline([
            ("anova", SelectKBest(score_func=f_classif, k=p.selected_features)),
            ("scaler", StandardScaler(copy=True, with_mean=True, with_std=True)),
            ("classifier", SVC(**dict(p.svm_parameters))),
        ], memory=None)
        with threadpool_limits(limits=1):
            pipeline.fit(population.X, population.y)
            selected = pipeline[:-1].transform(population.X)
            mahalanobis = PredictedClassMahalanobisScorer().fit(selected, population.y, identity_count)
        model = FittedV3Identity(pipeline, None, mahalanobis, identity_count, methods)
        validate_fitted_model_state(model, self.config, len(population.y))
        return model

    fingerprints = staticmethod(fitted_model_fingerprints)

    @staticmethod
    def support_vector_count(model):
        return int(model.pipeline[-1].n_support_.sum())

    @staticmethod
    def model_identity(model):
        return (id(model), id(model.pipeline), id(model.pipeline.named_steps["anova"]),
                id(model.pipeline.named_steps["scaler"]), id(model.pipeline[-1]), id(model.mahalanobis))

    def infer(self, model, population):
        with threadpool_limits(limits=1):
            Z = model.pipeline[:-1].transform(population.X)
            classifier = model.pipeline[-1]
            predicted = classifier.predict(Z)
            decisions = classifier.decision_function(Z)
            if model.identity_count == 2 and decisions.ndim == 1:
                decisions = np.column_stack((-decisions, decisions))
            if decisions.ndim != 2 or decisions.shape[1] != model.identity_count:
                raise RejectionError("SVM score columns do not match class ordering")
            values = model.mahalanobis.score(Z, predicted)
        return FrameScores(predicted, decisions, (("mahalanobis_distance", values),))
