"""Explicit feature-transform pipelines using reusable sklearn classifiers."""

import numpy as np
from sklearn.pipeline import Pipeline

from face_recognition.domain.models.classification import ClassifierConfiguration
from face_recognition.domain.models.feature_transforms import FittedReductionMetadata, ReducedMeasurements, ReductionConfiguration
from face_recognition.infrastructure.classifiers.sklearn_classification import SklearnClassifierRunner, build_classifier_pipeline
from face_recognition.infrastructure.features.feature_transformers import feature_transform_steps, fitted_reduction_metadata


def build_reduction_pipeline(configuration: ClassifierConfiguration, reduction: ReductionConfiguration,
                             original_features: int) -> Pipeline:
    baseline = build_classifier_pipeline(configuration)
    if reduction.method == "baseline":
        return baseline
    steps = feature_transform_steps(reduction, original_features)
    # PCA already includes the only scaler; ANOVA precedes the normal baseline
    # scaler when present. RF receives scaling only as required before PCA.
    steps.extend(baseline.steps[-1:] if reduction.method == "pca" else baseline.steps)
    return Pipeline(steps, memory=None)


class SklearnFeatureTransformRunner(SklearnClassifierRunner):
    def run_reduced(self, reduction: ReductionConfiguration, configuration: ClassifierConfiguration,
                    X_fit: np.ndarray, y_fit: np.ndarray, X_validation: np.ndarray, y_validation: np.ndarray,
                    identity_count: int) -> ReducedMeasurements:
        self.validate_inputs(configuration, X_fit, y_fit, X_validation, y_validation)
        if reduction.method == "baseline":
            measurements = self.run(configuration, X_fit, y_fit, X_validation, y_validation, identity_count)
            return ReducedMeasurements(measurements, FittedReductionMetadata(X_fit.shape[1]))
        model = build_reduction_pipeline(configuration, reduction, X_fit.shape[1])
        measurements = self.evaluate_pipeline(model, X_fit, y_fit, X_validation, y_validation, identity_count)
        metadata = fitted_reduction_metadata(model, reduction, X_fit.shape[1])
        return ReducedMeasurements(measurements, metadata)
