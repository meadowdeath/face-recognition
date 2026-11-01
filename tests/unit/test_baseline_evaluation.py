from dataclasses import replace
from itertools import count
import unittest
from unittest.mock import Mock, patch

import numpy as np
from sklearn.preprocessing import StandardScaler

from face_recognition.application.evaluate_baseline_models import BaselineEvaluationError, BaselineIdentityEvaluator, summarize_classifier
from face_recognition.application.training_folds import create_session_folds
from face_recognition.domain.models.baseline_evaluation import ClassificationMetrics, EvaluationTiming
from face_recognition.domain.models.identity_split import IdentityPartition
from face_recognition.infrastructure.classifiers.sklearn_baselines import (
    BASELINE_CONFIGURATIONS, SklearnBaselineRunner, build_baseline_pipeline, classification_measurements,
)
from tests.baseline_fakes import development_data


class BaselineEvaluationTests(unittest.TestCase):
    def evaluate(self, data=None):
        return BaselineIdentityEvaluator(SklearnBaselineRunner()).evaluate(data or development_data())

    def test_three_classifiers_evaluate_two_whole_session_folds(self):
        report = self.evaluate()
        self.assertEqual((report.representation, report.feature_count, report.identity_count, report.development_samples),
                         ("all_xyz_v1", 9, 2, 24))
        self.assertEqual(report.label_order, ("known_alpha", "known_beta"))
        self.assertEqual(tuple(row.configuration.name for row in report.classifiers), ("KNN", "SVM", "Random Forest"))
        for classifier in report.classifiers:
            with self.subTest(classifier=classifier.configuration.name):
                self.assertEqual(tuple(f.fold_name for f in classifier.folds), ("Fold A", "Fold B"))
                for fold in classifier.folds:
                    self.assertEqual((fold.fit_samples, fold.validation_samples, fold.feature_count), (12, 12, 9))
                    self.assertFalse(set(fold.fit_session_ids) & set(fold.validation_session_ids))
                    self.assertEqual(fold.label_order, report.label_order)
                    self.assertEqual(fold.measurements.metrics, ClassificationMetrics(1, 1, 1, 1))
                    self.assertEqual(fold.measurements.confusion_matrix, ((6, 0), (0, 6)))
                self.assertEqual(classifier.mean_metrics, ClassificationMetrics(1, 1, 1, 1))
                self.assertEqual(classifier.std_metrics, ClassificationMetrics(0, 0, 0, 0))

    def test_scaler_is_fitted_only_on_each_fit_side_and_sources_are_unchanged(self):
        data = development_data()
        before = data.X.tobytes()
        expected = tuple(fold.fit.X.mean(axis=0, dtype=np.float64) for fold in create_session_folds(data))
        observed = []
        original_fit = StandardScaler.fit

        def fit(scaler, X, y=None, sample_weight=None):
            self.assertTrue(any(np.shares_memory(X, fold.fit.X) for fold in create_session_folds(data)))
            original_fit(scaler, X, y, sample_weight=sample_weight)
            observed.append((int(scaler.n_samples_seen_), scaler.mean_.copy()))
            return scaler

        with patch.object(StandardScaler, "fit", fit):
            self.evaluate(data)
        self.assertEqual(len(observed), 4)  # KNN and SVM, once for each fold. RF never fits a scaler.
        for (sample_count, mean), expected_mean in zip(observed, (*expected, *expected)):
            self.assertEqual(sample_count, 12)
            self.assertTrue(np.allclose(mean, expected_mean), "Scaler statistics must come only from the fit session")
        self.assertFalse(np.allclose(expected[0], data.X.mean(axis=0, dtype=np.float64)))
        self.assertTrue(before == data.X.tobytes(), "Evaluation must not modify source feature values")

    def test_random_forest_pipeline_has_no_scaler_and_baselines_are_fixed(self):
        knn, svm, forest = (build_baseline_pipeline(spec) for spec in BASELINE_CONFIGURATIONS)
        self.assertEqual(tuple(knn.named_steps), ("scaler", "classifier"))
        self.assertEqual(tuple(svm.named_steps), ("scaler", "classifier"))
        self.assertEqual(tuple(forest.named_steps), ("classifier",))
        self.assertEqual(knn.named_steps["classifier"].n_neighbors, 5)
        self.assertEqual((svm.named_steps["classifier"].kernel, svm.named_steps["classifier"].C,
                          svm.named_steps["classifier"].gamma), ("rbf", 1.0, "scale"))
        self.assertEqual((forest.named_steps["classifier"].n_estimators, forest.named_steps["classifier"].random_state,
                          forest.named_steps["classifier"].n_jobs), (100, 42, 1))

    def test_metrics_and_confusion_matrix_use_all_labels_and_zero_division_zero(self):
        actual, matrix = classification_measurements(np.array([0, 0, 1, 1, 2, 2]), np.array([0, 1, 1, 1, 0, 1]), 3)
        self.assertEqual(matrix, ((1, 1, 0), (0, 2, 0), (1, 1, 0)))
        self.assertAlmostEqual(actual.accuracy, 0.5)
        self.assertAlmostEqual(actual.macro_precision, 1 / 3)
        self.assertAlmostEqual(actual.macro_recall, 0.5)
        self.assertAlmostEqual(actual.macro_f1, 7 / 18)

    def test_population_fold_summary_is_unweighted_and_keeps_matrices_separate(self):
        classifier = self.evaluate().classifiers[0]
        first, second = classifier.folds
        first = replace(first, measurements=replace(first.measurements, metrics=ClassificationMetrics(.2, .3, .4, .5),
                                                    timing=EvaluationTiming(2, 4, 6)))
        second = replace(second, measurements=replace(second.measurements, metrics=ClassificationMetrics(.6, .7, .8, .9),
                                                      timing=EvaluationTiming(4, 6, 8)))
        summary = summarize_classifier(classifier.configuration, (first, second))
        for name, expected in zip(ClassificationMetrics.__dataclass_fields__, (.4, .5, .6, .7)):
            self.assertAlmostEqual(getattr(summary.mean_metrics, name), expected)
            self.assertAlmostEqual(getattr(summary.std_metrics, name), .2)
        self.assertEqual(summary.mean_timing, EvaluationTiming(3, 5, 7))
        self.assertEqual(summary.std_timing, EvaluationTiming(1, 1, 1))
        self.assertEqual(len(summary.folds), 2)

    def test_deterministic_metrics_and_matrices_but_not_real_clock_values(self):
        first, second = self.evaluate(), self.evaluate()
        for left, right in zip(first.classifiers, second.classifiers):
            self.assertEqual(left.configuration, right.configuration)
            self.assertEqual(tuple((f.measurements.metrics, f.measurements.confusion_matrix) for f in left.folds),
                             tuple((f.measurements.metrics, f.measurements.confusion_matrix) for f in right.folds))

    def test_fresh_pipeline_created_for_each_fold_sequentially(self):
        events = []
        original = build_baseline_pipeline

        def build(spec):
            model = original(spec)
            events.append((spec.name, id(model)))
            return model

        with patch("face_recognition.infrastructure.classifiers.sklearn_baselines.build_baseline_pipeline", build):
            self.evaluate()
        self.assertEqual(tuple(name for name, _ in events), ("KNN", "KNN", "SVM", "SVM", "Random Forest", "Random Forest"))

    def test_timing_arithmetic_with_a_synthetic_clock(self):
        runner = SklearnBaselineRunner(clock=iter((10.0, 12.0, 12.5)).__next__)
        fold = create_session_folds(development_data())[0]
        result = runner.run(BASELINE_CONFIGURATIONS[0], *fold.fit.arrays(), *fold.validation.arrays(), 2)
        self.assertEqual(result.timing, EvaluationTiming(2.0, .5, 500 / 12))

    def test_arbitrary_dimensions_preserve_all_columns_without_selection(self):
        report = self.evaluate(development_data(feature_count=21, samples_per_identity=7))
        self.assertEqual(report.feature_count, 21)
        self.assertTrue(all(fold.feature_count == 21 for row in report.classifiers for fold in row.folds))

    def test_inconsistent_fit_validation_feature_dimensions_are_rejected(self):
        data = development_data()
        with self.assertRaisesRegex(ValueError, "dimensions"):
            SklearnBaselineRunner().run(BASELINE_CONFIGURATIONS[0], data.X[:12], data.y[:12], data.X[12:, :-1], data.y[12:], 2)

    def test_nonfinite_features_are_rejected_before_any_model_fit_without_values_in_error(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            data = development_data()
            data.X.setflags(write=True)
            data.X[0, 0] = value
            runner = Mock()
            with self.assertRaises(BaselineEvaluationError) as caught:
                BaselineIdentityEvaluator(runner).evaluate(data)
            runner.run.assert_not_called()
            self.assertNotIn(str(value), str(caught.exception))

    def test_protected_partitions_are_rejected_before_model_fit(self):
        for partition in (IdentityPartition.KNOWN_TEST, IdentityPartition.UNKNOWN_POOL):
            data = development_data()
            object.__setattr__(data, "samples", (replace(data.samples[0], partition=partition), *data.samples[1:]))
            runner = Mock()
            with self.assertRaises(BaselineEvaluationError):
                BaselineIdentityEvaluator(runner).evaluate(data)
            runner.run.assert_not_called()

    def test_mismatched_identity_labels_and_zero_feature_count_are_rejected(self):
        for kind in ("labels", "features"):
            data = development_data()
            if kind == "labels":
                data.y.setflags(write=True)
                data.y[0] = 1
            else:
                object.__setattr__(data, "X", data.X[:, :0])
            with self.assertRaises(BaselineEvaluationError):
                self.evaluate(data)

    def test_model_exception_contents_are_sanitized(self):
        marker = "SYNTHETIC_PRIVATE_VECTOR_PAYLOAD"
        runner = SklearnBaselineRunner()
        with patch.object(runner, "run", side_effect=RuntimeError(marker)):
            with self.assertRaisesRegex(BaselineEvaluationError, "KNN, Fold A") as caught:
                BaselineIdentityEvaluator(runner).evaluate(development_data())
        self.assertNotIn(marker, str(caught.exception))

    def test_knn_does_not_silently_reduce_neighbor_count_for_small_fit_sets(self):
        with self.assertRaises(BaselineEvaluationError):
            self.evaluate(development_data(samples_per_identity=2))
