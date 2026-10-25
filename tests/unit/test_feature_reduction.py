from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from face_recognition.application.evaluate_baseline_models import BaselineIdentityEvaluator
from face_recognition.application.evaluate_feature_reduction import FeatureReductionError, FeatureReductionEvaluator, rank_results
from face_recognition.domain.models.baseline_evaluation import ClassificationMetrics, EvaluationTiming
from face_recognition.domain.models.feature_reduction import DEFAULT_REDUCTIONS, ReductionConfiguration
from face_recognition.domain.models.identity_split import IdentityPartition
from face_recognition.infrastructure.classifiers.sklearn_baselines import SklearnBaselineRunner
from face_recognition.infrastructure.classifiers.sklearn_feature_reduction import SklearnFeatureReductionRunner
from tests.feature_reduction_fakes import SMALL_REDUCTIONS, RecordingReductionRunner, reduction_data


class FeatureReductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = reduction_data()
        cls.report = FeatureReductionEvaluator(SklearnFeatureReductionRunner(), SMALL_REDUCTIONS).evaluate(cls.data)

    def test_actual_anova_pca_and_three_classifiers_reuse_two_session_folds(self):
        self.assertEqual(len(self.report.results), len(SMALL_REDUCTIONS) * 3)
        for result in self.report.results:
            self.assertEqual(len(result.evaluation.folds), 2)
            for fold, metadata in zip(result.evaluation.folds, result.transformations):
                self.assertEqual(fold.feature_count, metadata.transformation.retained_features)
                self.assertEqual((fold.fit_samples, fold.validation_samples), (16, 16))
                self.assertFalse(set(fold.fit_session_ids) & set(fold.validation_session_ids))
                self.assertEqual(sum(map(sum, fold.measurements.confusion_matrix)), 16)
            metrics = result.evaluation.mean_metrics
            for name in metrics.__dataclass_fields__:
                self.assertAlmostEqual(getattr(metrics, name),
                                       sum(getattr(row.measurements.metrics, name) for row in result.evaluation.folds) / 2)

    def test_all_production_candidates_and_models_are_present_in_sequential_order(self):
        runner = RecordingReductionRunner()
        report = FeatureReductionEvaluator(runner).evaluate(reduction_data(feature_count=1026))
        self.assertEqual([row.k for row in DEFAULT_REDUCTIONS if row.method == "anova"], [256, 512, 768, 1024])
        self.assertEqual([row.explained_variance for row in DEFAULT_REDUCTIONS if row.method == "pca"], [.90, .95, .99])
        self.assertEqual(len(report.results), 24)
        self.assertEqual(len(runner.calls), 48)
        expected = [(reduction.identifier, classifier.name) for reduction in DEFAULT_REDUCTIONS
                    for classifier in runner.configurations for _ in range(2)]
        self.assertEqual([(identifier, name) for identifier, name, _, _ in runner.calls], expected)

    def test_baseline_metrics_and_confusion_matrices_match_original_evaluator(self):
        original = BaselineIdentityEvaluator(SklearnBaselineRunner()).evaluate(self.data)
        for result, expected in zip(self.report.results[:3], original.classifiers):
            self.assertEqual(result.reduction.method, "baseline")
            self.assertEqual(result.evaluation.configuration, expected.configuration)
            self.assertEqual(result.evaluation.mean_metrics, expected.mean_metrics)
            self.assertEqual(tuple(f.measurements.confusion_matrix for f in result.evaluation.folds),
                             tuple(f.measurements.confusion_matrix for f in expected.folds))
            self.assertEqual(result.mean_retained_features, 9)
            self.assertEqual(result.comparison.feature_count_reduction, 0)
            self.assertEqual(result.comparison.macro_f1_delta, 0)

    def test_comparisons_use_matching_classifier_and_correct_feature_reduction(self):
        references = {row.evaluation.configuration.name: row for row in self.report.results if row.reduction.method == "baseline"}
        for row in self.report.results:
            baseline = references[row.evaluation.configuration.name]
            self.assertAlmostEqual(row.comparison.macro_f1_delta,
                                   row.evaluation.mean_metrics.macro_f1 - baseline.evaluation.mean_metrics.macro_f1)
            self.assertAlmostEqual(row.comparison.accuracy_delta,
                                   row.evaluation.mean_metrics.accuracy - baseline.evaluation.mean_metrics.accuracy)
            self.assertAlmostEqual(row.comparison.prediction_latency_delta_ms,
                                   row.evaluation.mean_timing.mean_prediction_latency_ms - baseline.evaluation.mean_timing.mean_prediction_latency_ms)
            self.assertAlmostEqual(row.comparison.feature_count_reduction, 9 - row.mean_retained_features)
            self.assertAlmostEqual(row.comparison.feature_count_reduction_percent, 100 * (9 - row.mean_retained_features) / 9)
            if row.reduction.method == "anova":
                self.assertEqual(row.mean_retained_features, row.reduction.k)

    def test_pca_does_not_force_equal_fold_component_counts(self):
        data = reduction_data()
        X = data.X.copy()
        X[:16] = np.outer(np.arange(16, dtype=np.float32), np.arange(1, 10, dtype=np.float32))
        data = replace(data, X=X)
        report = FeatureReductionEvaluator(SklearnFeatureReductionRunner(),
                                          (ReductionConfiguration("baseline"), ReductionConfiguration("pca", explained_variance=.95))).evaluate(data)
        for row in report.results[3:]:
            dimensions = tuple(f.transformation.retained_features for f in row.transformations)
            self.assertNotEqual(*dimensions)
            self.assertEqual(row.mean_retained_features, sum(dimensions) / 2)

    def test_ranking_uses_requested_priorities_and_stable_final_ties(self):
        original = self.report.results[0]

        def row(identifier, f1, accuracy, dimension, latency):
            evaluation = replace(original.evaluation, mean_metrics=ClassificationMetrics(accuracy, .5, .5, f1),
                                 mean_timing=EvaluationTiming(1, 1, latency))
            return replace(original, reduction=ReductionConfiguration("anova", k=identifier), evaluation=evaluation,
                           mean_retained_features=dimension)

        rows = (row(1, .9, .8, 4, 2), row(2, .9, .8, 4, 1), row(3, .9, .8, 3, 20),
                row(4, .9, .9, 9, 30), row(5, .95, .5, 9, 30), row(6, .9, .8, 4, 1))
        ranking = rank_results(rows)
        self.assertEqual([item.reduction_id for item in ranking], ["anova_k_5", "anova_k_4", "anova_k_3", "anova_k_2", "anova_k_6", "anova_k_1"])
        self.assertEqual(ranking, rank_results(tuple(reversed(rows))))

    def test_invalid_or_oversized_counts_fail_without_silently_skipping(self):
        runner = RecordingReductionRunner()
        with self.assertRaisesRegex(FeatureReductionError, "exceeds"):
            FeatureReductionEvaluator(runner).evaluate(reduction_data())
        self.assertEqual(runner.calls, [])
        for reductions in ((), (ReductionConfiguration("anova", k=2),),
                           (ReductionConfiguration("baseline"), ReductionConfiguration("baseline"))):
            with self.assertRaises(FeatureReductionError):
                FeatureReductionEvaluator(runner, reductions)

    def test_nonfinite_and_protected_data_fail_before_any_model_call(self):
        for partition in (IdentityPartition.KNOWN_TEST, IdentityPartition.UNKNOWN_POOL, None):
            data = reduction_data()
            if partition is None:
                data.X.setflags(write=True)
                data.X[0, 0] = float("nan")
            else:
                object.__setattr__(data, "samples", (replace(data.samples[0], partition=partition), *data.samples[1:]))
            runner = RecordingReductionRunner()
            with self.assertRaises(FeatureReductionError):
                FeatureReductionEvaluator(runner, SMALL_REDUCTIONS).evaluate(data)
            self.assertEqual(runner.calls, [])

    def test_backend_errors_never_expose_private_values(self):
        runner = RecordingReductionRunner()
        with patch.object(runner, "run_reduced", side_effect=ValueError("SYNTHETIC_PRIVATE_VECTOR")):
            with self.assertRaises(FeatureReductionError) as caught:
                FeatureReductionEvaluator(runner, SMALL_REDUCTIONS).evaluate(reduction_data())
        self.assertNotIn("SYNTHETIC_PRIVATE_VECTOR", str(caught.exception))
