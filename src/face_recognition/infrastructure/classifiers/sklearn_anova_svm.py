"""Sequential ANOVA-SVM fold execution using existing builders and metrics."""

from dataclasses import replace

import numpy as np

from face_recognition.domain.models.anova_svm_tuning import AnovaSvmCandidate, SupportVectorCounts, TuningFoldMeasurements
from face_recognition.domain.models.baseline_evaluation import BaselineModelConfiguration
from face_recognition.domain.models.feature_reduction import ReductionConfiguration
from face_recognition.infrastructure.classifiers.sklearn_baselines import BASELINE_CONFIGURATIONS, SklearnBaselineRunner
from face_recognition.infrastructure.classifiers.sklearn_feature_reduction import build_reduction_pipeline
from face_recognition.infrastructure.features.feature_transformers import fitted_reduction_metadata

SVM_BASELINE_CONFIGURATION = next(row for row in BASELINE_CONFIGURATIONS if row.name == "SVM")


class SklearnAnovaSvmRunner(SklearnBaselineRunner):
    # BaselineIdentityEvaluator uses only SVM for the full-feature reference.
    configurations = (SVM_BASELINE_CONFIGURATION,)

    @staticmethod
    def configuration_for(candidate: AnovaSvmCandidate) -> BaselineModelConfiguration:
        parameters = dict(SVM_BASELINE_CONFIGURATION.classifier_parameters)
        parameters.update(C=candidate.C, gamma=candidate.gamma)
        return replace(SVM_BASELINE_CONFIGURATION, classifier_parameters=tuple(parameters.items()))

    def run_candidate(self, candidate: AnovaSvmCandidate, X_fit: np.ndarray, y_fit: np.ndarray,
                      X_validation: np.ndarray, y_validation: np.ndarray, identity_count: int) -> TuningFoldMeasurements:
        configuration = self.configuration_for(candidate)
        self.validate_inputs(configuration, X_fit, y_fit, X_validation, y_validation)
        reduction = ReductionConfiguration("anova", k=candidate.k)
        model = build_reduction_pipeline(configuration, reduction, X_fit.shape[1])
        measurements = self.evaluate_pipeline(model, X_fit, y_fit, X_validation, y_validation, identity_count)
        metadata = fitted_reduction_metadata(model, reduction, X_fit.shape[1])
        classifier = model.named_steps["classifier"]
        if not np.array_equal(classifier.classes_, np.arange(identity_count)):
            raise ValueError("SVM class order differs from the development label mapping")
        counts = tuple(int(value) for value in classifier.n_support_)
        total = int(classifier.support_.size)
        if total > len(y_fit):
            raise ValueError("SVM support count exceeds fit sample count")
        return TuningFoldMeasurements(measurements, metadata.selected_feature_indices, SupportVectorCounts(total, counts))
