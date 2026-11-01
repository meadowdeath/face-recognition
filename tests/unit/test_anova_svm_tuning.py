from dataclasses import replace
from statistics import fmean, pstdev
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from face_recognition.application.training_folds import create_session_folds
from face_recognition.application.tune_anova_svm import AnovaSvmTuner, AnovaSvmTuningError, rank_tuning_results, selection_stability
from face_recognition.domain.models.anova_svm_tuning import (
    ANOVA_K_VALUES, REFERENCE_CANDIDATE, SVM_C_VALUES, SVM_GAMMA_VALUES, AnovaSvmCandidate, tuning_candidates,
)
from face_recognition.domain.models.baseline_evaluation import ClassificationMetrics, EvaluationTiming
from face_recognition.domain.models.feature_reduction import ReductionConfiguration
from face_recognition.domain.models.identity_split import IdentityPartition
from face_recognition.infrastructure.classifiers.sklearn_anova_svm import SVM_BASELINE_CONFIGURATION, SklearnAnovaSvmRunner
from face_recognition.infrastructure.classifiers.sklearn_feature_reduction import SklearnFeatureReductionRunner, build_reduction_pipeline
from tests.feature_reduction_fakes import reduction_data


class AnovaSvmTuningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = reduction_data(feature_count=768)
        cls.report = AnovaSvmTuner(SklearnAnovaSvmRunner()).evaluate(cls.data)

    def test_exact_fixed_space_order_and_reference_inclusion(self):
        candidates = tuning_candidates()
        self.assertEqual(ANOVA_K_VALUES, (384, 448, 512, 640, 768))
        self.assertEqual(SVM_C_VALUES, (.5, 1.0, 2.0, 4.0))
        self.assertEqual(SVM_GAMMA_VALUES, ("scale", .001, .01))
        self.assertEqual(len(candidates), 60)
        self.assertEqual(len({row.identifier for row in candidates}), 60)
        self.assertEqual(candidates, tuning_candidates())
        self.assertEqual([(row.k, row.C, row.gamma) for row in candidates],
                         [(k, C, gamma) for k in ANOVA_K_VALUES for C in SVM_C_VALUES for gamma in SVM_GAMMA_VALUES])
        self.assertEqual(candidates.count(REFERENCE_CANDIDATE), 1)
        self.assertTrue(all(row.kernel == "rbf" and row.probability is False for row in candidates))

    def test_outside_grid_and_probability_variants_are_rejected(self):
        for changes in ({"k": 256}, {"C": 8}, {"gamma": .1}, {"gamma": "auto"},
                        {"kernel": "linear"}, {"probability": True}, {"C": True}, {"k": 512.0}):
            with self.subTest(field=next(iter(changes))), self.assertRaises(ValueError):
                replace(REFERENCE_CANDIDATE, **changes)

    def test_only_C_and_gamma_change_from_existing_svm_configuration(self):
        runner = SklearnAnovaSvmRunner()
        baseline = dict(SVM_BASELINE_CONFIGURATION.classifier_parameters)
        for candidate in tuning_candidates():
            actual = runner.configuration_for(candidate)
            expected = {**baseline, "C": candidate.C, "gamma": candidate.gamma}
            self.assertEqual(dict(actual.classifier_parameters), expected)
            self.assertEqual(actual.scaler_parameters, SVM_BASELINE_CONFIGURATION.scaler_parameters)
        self.assertEqual(runner.configuration_for(REFERENCE_CANDIDATE), SVM_BASELINE_CONFIGURATION)

    def test_all_candidates_two_session_folds_correct_metrics_aggregation_and_support_counts(self):
        self.assertEqual(tuple(row.candidate for row in self.report.results), tuning_candidates())
        self.assertEqual(len(self.report.ranking), 60)
        for row in self.report.results:
            folds = row.evaluation.folds
            self.assertEqual(len(folds), 2)
            for fold, diagnostic in zip(folds, row.support_vectors):
                self.assertEqual((fold.feature_count, fold.fit_samples, fold.validation_samples), (row.candidate.k, 16, 16))
                self.assertFalse(set(fold.fit_session_ids) & set(fold.validation_session_ids))
                self.assertEqual(sum(map(sum, fold.measurements.confusion_matrix)), 16)
                self.assertEqual(fold.label_order, self.data.label_to_person)
                self.assertEqual(diagnostic.counts.total, sum(diagnostic.counts.per_class))
                self.assertEqual(len(diagnostic.counts.per_class), 2)
                self.assertLessEqual(diagnostic.counts.total, fold.fit_samples)
            f1 = [fold.measurements.metrics.macro_f1 for fold in folds]
            self.assertEqual(row.worst_fold_macro_f1, min(f1))
            self.assertEqual(row.fold_f1_gap, abs(f1[0] - f1[1]))
            for name in ClassificationMetrics.__dataclass_fields__:
                values = [getattr(fold.measurements.metrics, name) for fold in folds]
                self.assertAlmostEqual(getattr(row.evaluation.mean_metrics, name), fmean(values))
                self.assertAlmostEqual(getattr(row.evaluation.std_metrics, name), pstdev(values))

    def test_reference_matches_previous_anova_svm_execution(self):
        row = next(row for row in self.report.results if row.candidate == REFERENCE_CANDIDATE)
        runner = SklearnFeatureReductionRunner()
        for fold, expected in zip(create_session_folds(self.data), row.evaluation.folds):
            result = runner.run_reduced(ReductionConfiguration("anova", k=512), SVM_BASELINE_CONFIGURATION,
                                        *fold.fit.arrays(), *fold.validation.arrays(), 2)
            self.assertEqual(result.measurements.metrics, expected.measurements.metrics)
            self.assertEqual(result.measurements.confusion_matrix, expected.measurements.confusion_matrix)

    def test_comparisons_reference_matching_full_and_anova512_metrics_and_dimensions(self):
        full = self.report.full_feature_svm_reference
        reference = next(row for row in self.report.results if row.candidate == REFERENCE_CANDIDATE)
        for row in self.report.results:
            for actual, expected, dimension in ((row.vs_full_svm_baseline, full, 768),
                                                 (row.vs_anova512_reference, reference.evaluation, 512)):
                self.assertAlmostEqual(actual.macro_f1_delta, row.evaluation.mean_metrics.macro_f1 - expected.mean_metrics.macro_f1)
                self.assertAlmostEqual(actual.accuracy_delta, row.evaluation.mean_metrics.accuracy - expected.mean_metrics.accuracy)
                self.assertEqual(actual.feature_count_reduction, dimension - row.candidate.k)
                self.assertAlmostEqual(actual.feature_count_reduction_percent, 100 * (dimension - row.candidate.k) / dimension)
                self.assertAlmostEqual(actual.prediction_latency_delta_ms,
                                       row.evaluation.mean_timing.mean_prediction_latency_ms - expected.mean_timing.mean_prediction_latency_ms)
        self.assertEqual(reference.vs_anova512_reference.macro_f1_delta, 0)
        self.assertEqual(reference.vs_anova512_reference.prediction_latency_delta_ms, 0)

    def test_known_overlap_statistics_and_invalid_indices(self):
        actual = selection_stability(4, (0, 1, 2, 3), (2, 3, 4, 5))
        self.assertEqual(actual.shared_features, 2)
        self.assertEqual(actual.overlap_percent, 50)
        self.assertAlmostEqual(actual.jaccard_similarity, 1 / 3)
        self.assertEqual(selection_stability(2, (0, 1), (0, 1)).overlap_percent, 100)
        self.assertEqual(selection_stability(2, (0, 1), (2, 3)).overlap_percent, 0)
        for indices in ((1, 0), (0, 0), (-1, 0), (0,)):
            with self.assertRaises(AnovaSvmTuningError):
                selection_stability(2, indices, (0, 1))
        self.assertEqual(tuple(row.k for row in self.report.selection_stability), ANOVA_K_VALUES)

    def test_ranking_respects_every_requested_priority_and_final_ties(self):
        original = self.report.results[0]

        def result(k, C, gamma, mean=.8, worst=.6, gap=.4, accuracy=.8, latency=1):
            return replace(original, candidate=AnovaSvmCandidate(k, C, gamma), worst_fold_macro_f1=worst, fold_f1_gap=gap,
                           evaluation=replace(original.evaluation, mean_metrics=ClassificationMetrics(accuracy, .8, .8, mean),
                                              mean_timing=EvaluationTiming(1, 1, latency)))

        # Synthetic independent diagnostic fields isolate each comparator.
        rows = (result(512, .5, "scale", mean=.9), result(512, 1, "scale", worst=.7),
                result(512, 2, "scale", gap=.3), result(512, 4, "scale", accuracy=.9),
                result(384, .5, "scale", latency=10), result(512, .5, .001, latency=.5),
                result(512, 1, .001), result(512, 1, .01))
        ranked = rank_tuning_results(tuple(reversed(rows)))
        self.assertEqual([row.candidate_id for row in ranked], [row.candidate.identifier for row in rows])
        self.assertEqual(ranked, rank_tuning_results(rows))
        self.assertEqual(self.report.selected_development_candidate.identifier, self.report.ranking[0].candidate_id)

    def test_selector_scaler_svm_fit_only_fit_data_and_diagnostics_match_sklearn(self):
        fold = create_session_folds(self.data)[0]
        X, y = fold.fit.arrays()
        observed, models = {"selector": [], "scaler": [], "svm": []}, []
        originals = SelectKBest.fit, StandardScaler.fit, SVC.fit

        def selector_fit(model, values, labels=None):
            observed["selector"].append((values.copy(), labels.copy()))
            return originals[0](model, values, labels)

        def scaler_fit(model, values, labels=None, **kwargs):
            observed["scaler"].append(values.copy())
            return originals[1](model, values, labels, **kwargs)

        def svm_fit(model, values, labels, **kwargs):
            observed["svm"].append((values.copy(), labels.copy()))
            return originals[2](model, values, labels, **kwargs)

        def build(*args):
            model = build_reduction_pipeline(*args)
            models.append(model)
            return model

        before = self.data.X.tobytes()
        with patch.object(SelectKBest, "fit", selector_fit), patch.object(StandardScaler, "fit", scaler_fit), \
             patch.object(SVC, "fit", svm_fit), \
             patch("face_recognition.infrastructure.classifiers.sklearn_anova_svm.build_reduction_pipeline", build):
            runner = SklearnAnovaSvmRunner()
            first = runner.run_candidate(REFERENCE_CANDIDATE, X, y, *fold.validation.arrays(), 2)
            second = runner.run_candidate(REFERENCE_CANDIDATE, X, y, fold.validation.X + 100, fold.validation.y, 2)
        self.assertEqual([len(rows) for rows in observed.values()], [2, 2, 2])
        for index, model in enumerate(models):
            self.assertTrue(np.array_equal(observed["selector"][index][0], X))
            self.assertTrue(np.array_equal(observed["selector"][index][1], y))
            self.assertTrue(np.array_equal(observed["scaler"][index], model.named_steps["anova"].transform(X)))
            self.assertTrue(np.allclose(observed["svm"][index][0], model[:-1].transform(X)))
            self.assertTrue(np.array_equal(observed["svm"][index][1], y))
            self.assertEqual(int(model.named_steps["scaler"].n_samples_seen_), len(y))
        self.assertEqual(first.selected_feature_indices, second.selected_feature_indices)
        self.assertEqual(first.support_vectors.per_class, tuple(int(value) for value in models[0].named_steps["classifier"].n_support_))
        self.assertEqual(first.support_vectors.total, len(models[0].named_steps["classifier"].support_))
        self.assertTrue(np.array_equal(models[0].named_steps["classifier"].classes_, np.arange(2)))
        self.assertTrue(before == self.data.X.tobytes(), "Source features remain unchanged")

    def test_invalid_source_and_protected_partitions_rejected_before_fitting(self):
        for invalid in ("small", "nonfinite", IdentityPartition.KNOWN_TEST, IdentityPartition.UNKNOWN_POOL):
            data = reduction_data(feature_count=9 if invalid == "small" else 768)
            if invalid == "nonfinite":
                data.X.setflags(write=True)
                data.X[0, 0] = float("nan")
            elif isinstance(invalid, IdentityPartition):
                object.__setattr__(data, "samples", (replace(data.samples[0], partition=invalid), *data.samples[1:]))
            runner = SklearnAnovaSvmRunner()
            with patch.object(runner, "run") as full_run, patch.object(runner, "run_candidate") as candidate_run:
                with self.assertRaises(AnovaSvmTuningError):
                    AnovaSvmTuner(runner).evaluate(data)
            full_run.assert_not_called()
            candidate_run.assert_not_called()

    def test_unchanged_selection_across_C_gamma_is_enforced(self):
        runner = SklearnAnovaSvmRunner()
        original = runner.run_candidate
        calls = []

        def run(*args):
            result = original(*args)
            calls.append(1)
            if len(calls) == 2:
                indices = result.selected_feature_indices
                other = next(index for index in range(768) if index not in indices)
                result = replace(result, selected_feature_indices=tuple(sorted((*indices[1:], other))))
            return result

        with patch.object(runner, "run_candidate", run), self.assertRaisesRegex(AnovaSvmTuningError, "changed across"):
            AnovaSvmTuner(runner).evaluate(self.data)

    def test_backend_errors_do_not_expose_feature_values(self):
        runner = SklearnAnovaSvmRunner()
        with patch.object(runner, "run_candidate", side_effect=ValueError("PRIVATE_SYNTHETIC_VECTOR")):
            with self.assertRaises(AnovaSvmTuningError) as caught:
                AnovaSvmTuner(runner).evaluate(self.data)
        self.assertNotIn("PRIVATE_SYNTHETIC_VECTOR", str(caught.exception))
