"""Reusable classifier primitives, with no archived experiment orchestration."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.preprocessing import StandardScaler

from face_recognition.application.training_folds import create_session_folds
from face_recognition.domain.models.classification import ClassificationMetrics, EvaluationTiming
from face_recognition.infrastructure.classifiers.sklearn_classification import (
    CLASSIFIER_CONFIGURATIONS, SklearnClassifierRunner, build_classifier_pipeline,
    class_decision_scores, classification_measurements,
)
from tests.classification_fakes import development_data


class SklearnClassificationTests(unittest.TestCase):
    def test_fresh_explicit_pipelines_and_conservative_classifier_parameters(self):
        knn, svm, forest = (build_classifier_pipeline(spec) for spec in CLASSIFIER_CONFIGURATIONS)
        self.assertEqual(tuple(knn.named_steps), ("scaler", "classifier"))
        self.assertEqual(tuple(svm.named_steps), ("scaler", "classifier"))
        self.assertEqual(tuple(forest.named_steps), ("classifier",))
        self.assertEqual(knn[-1].n_neighbors, 5)
        self.assertEqual((svm[-1].kernel, svm[-1].C, svm[-1].gamma, svm[-1].probability), ("rbf", 1., "scale", False))
        self.assertEqual((forest[-1].n_estimators, forest[-1].random_state, forest[-1].n_jobs), (100, 42, 1))
        self.assertIsNot(knn[-1], build_classifier_pipeline(CLASSIFIER_CONFIGURATIONS[0])[-1])

    def test_all_classifiers_execute_on_synthetic_fit_validation_matrices(self):
        fold = create_session_folds(development_data())[0]
        runner = SklearnClassifierRunner()
        for configuration in CLASSIFIER_CONFIGURATIONS:
            result = runner.run(configuration, *fold.fit.arrays(), *fold.validation.arrays(), 2)
            self.assertEqual(result.metrics, ClassificationMetrics(1., 1., 1., 1.))
            self.assertEqual(result.confusion_matrix, ((6, 0), (0, 6)))

    def test_scaler_fits_fit_rows_only_and_does_not_modify_source(self):
        fold = create_session_folds(development_data())[0]
        X, y = fold.fit.arrays()
        validation, validation_y = fold.validation.arrays()
        before = X.tobytes(), validation.tobytes()
        observed = []
        original = StandardScaler.fit

        def fit(scaler, values, labels=None, sample_weight=None):
            observed.append(values.copy())
            return original(scaler, values, labels, sample_weight=sample_weight)

        with patch.object(StandardScaler, "fit", fit):
            SklearnClassifierRunner().run(CLASSIFIER_CONFIGURATIONS[1], X, y, validation, validation_y, 2)
        self.assertEqual(len(observed), 1)
        np.testing.assert_array_equal(observed[0], X)
        self.assertEqual(before, (X.tobytes(), validation.tobytes()))

    def test_macro_metrics_confusion_and_zero_support_use_all_labels(self):
        actual, matrix = classification_measurements(np.array([0, 0, 1, 1, 2, 2]), np.array([0, 1, 1, 1, 0, 1]), 3)
        self.assertEqual(matrix, ((1, 1, 0), (0, 2, 0), (1, 1, 0)))
        self.assertEqual(actual.accuracy, .5)
        self.assertAlmostEqual(actual.macro_precision, 1 / 3)
        self.assertAlmostEqual(actual.macro_recall, .5)
        self.assertAlmostEqual(actual.macro_f1, 7 / 18)

    def test_synthetic_clock_and_inconsistent_dimensions(self):
        fold = create_session_folds(development_data())[0]
        runner = SklearnClassifierRunner(clock=iter((10., 12., 12.5)).__next__)
        result = runner.run(CLASSIFIER_CONFIGURATIONS[0], *fold.fit.arrays(), *fold.validation.arrays(), 2)
        self.assertEqual(result.timing, EvaluationTiming(2., .5, 500 / 12))
        with self.assertRaisesRegex(ValueError, "dimensions"):
            runner.run(CLASSIFIER_CONFIGURATIONS[0], fold.fit.X, fold.fit.y, fold.validation.X[:, :-1], fold.validation.y, 2)
        with self.assertRaisesRegex(ValueError, "five"):
            runner.run(CLASSIFIER_CONFIGURATIONS[0], fold.fit.X[:2], fold.fit.y[:2], fold.validation.X, fold.validation.y, 2)

    def test_class_ordered_binary_and_multiclass_decision_scores(self):
        binary = SimpleNamespace(decision_function=lambda X: np.array([-2., 3.]))
        np.testing.assert_array_equal(class_decision_scores(binary, None, 2), [[2., -2.], [-3., 3.]])
        matrix = np.array([[1., 3., 2.], [4., 1., 0.]])
        multiclass = SimpleNamespace(decision_function=lambda X: matrix)
        self.assertIs(class_decision_scores(multiclass, None, 3), matrix)
        with self.assertRaisesRegex(ValueError, "class ordering"):
            class_decision_scores(multiclass, None, 4)

    def test_measurement_domain_validation(self):
        for value in (float("nan"), float("inf"), -1., 1.1):
            with self.assertRaises(ValueError):
                ClassificationMetrics(value, .5, .5, .5)
        for value in (float("nan"), float("inf"), -1.):
            with self.assertRaises(ValueError):
                EvaluationTiming(value, 1., 1.)

