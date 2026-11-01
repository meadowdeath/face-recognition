from dataclasses import replace
from pathlib import Path
from statistics import fmean, pstdev
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from face_recognition.application.evaluate_rejection_scores import RejectionScoreEvaluator
from face_recognition.application.evaluate_temporal_open_set import TemporalOpenSetEvaluator, rank_temporal_results
from face_recognition.application.rejection_metrics import evaluate_threshold
from face_recognition.application.rejection_score_analysis import analyze_scores
from face_recognition.application.temporal_windows import aggregate_windows, build_windows
from face_recognition.application.training_folds import create_session_folds
from face_recognition.application.unknown_rejection_split import UnknownRejectionError
from face_recognition.domain.models.identity_split import IdentityPartition
from face_recognition.domain.models.temporal_open_set import TEMPORAL_MODEL, TEMPORAL_SCORES, WINDOW_SIZES
from face_recognition.infrastructure.classifiers.sklearn_feature_reduction import build_reduction_pipeline
from face_recognition.infrastructure.classifiers.sklearn_rejection_scores import SklearnRejectionScoreRunner
from face_recognition.infrastructure.classifiers.sklearn_temporal_open_set import SklearnTemporalOpenSetRunner
from face_recognition.infrastructure.features.rejection_scores import PredictedClassCentroidScorer
from tests.unknown_rejection_fakes import rejection_fixture, load_rejection_fixture
from tests.unit.test_temporal_windows import references


class TemporalOpenSetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory(prefix="synthetic_temporal_evaluation_")
        cls.processed, cls.source, cls.partition, _, _ = rejection_fixture(Path(cls.temporary.name), samples=8)
        cls.known, cls.unknown = load_rejection_fixture(cls.processed, cls.source, cls.partition)
        cls.report = TemporalOpenSetEvaluator(SklearnTemporalOpenSetRunner()).evaluate(cls.known, cls.unknown)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_fixed_model_two_scores_four_sizes_two_folds(self):
        self.assertEqual(self.report.model, TEMPORAL_MODEL)
        self.assertEqual(TEMPORAL_MODEL.k, 448)
        self.assertEqual([(r.score_family, r.window_size) for r in self.report.results],
                         [(family, size) for family in TEMPORAL_SCORES for size in WINDOW_SIZES])
        self.assertTrue(all(len(row.folds) == 2 for row in self.report.results))
        self.assertEqual(self.report.configuration, SklearnTemporalOpenSetRunner().configuration_for(TEMPORAL_MODEL))

    def test_window_one_reproduces_existing_rejection_scores(self):
        reference = RejectionScoreEvaluator(SklearnRejectionScoreRunner()).evaluate(self.known, self.unknown)
        for row in self.report.results:
            if row.window_size != 1:
                continue
            original = next(r for r in reference.results if r.candidate == TEMPORAL_MODEL and r.score_family == row.score_family)
            self.assertEqual(row.mean_metrics, original.mean_metrics)
            for fold, previous in zip(row.folds, original.folds):
                self.assertEqual(fold.analysis, previous.analysis)
                self.assertEqual(fold.frame_closed_set_metrics, previous.closed_set_metrics)
                self.assertEqual(fold.known_vote_stability.rates.unanimous_rate, 1)

    def test_independent_calibration_per_fold_score_size(self):
        calls = []

        def analyze(known, correct, unknown):
            calls.append((len(known), len(correct), len(unknown)))
            return analyze_scores(known, correct, unknown)

        with patch("face_recognition.application.evaluate_temporal_open_set.analyze_scores", analyze):
            TemporalOpenSetEvaluator(SklearnTemporalOpenSetRunner()).evaluate(self.known, self.unknown)
        per_fold = [(48, 48, 32)] * 2 + [(12, 12, 8)] * 2 + [(6, 6, 4)] * 4
        self.assertEqual(calls, per_fold * 2)

    def test_used_discarded_counts_and_condition_totals(self):
        expected = {1: (48, 48, 0, 32, 32, 0), 3: (12, 36, 12, 8, 24, 8),
                    5: (6, 30, 18, 4, 20, 12), 7: (6, 42, 6, 4, 28, 4)}
        for row in self.report.results:
            kw, ku, kd, uw, uu, ud = expected[row.window_size]
            self.assertEqual((row.total_known_counts.windows, row.total_known_counts.used_samples,
                              row.total_known_counts.discarded_samples), (2 * kw, 2 * ku, 2 * kd))
            self.assertEqual((row.total_unknown_counts.windows, row.total_unknown_counts.used_samples,
                              row.total_unknown_counts.discarded_samples), (2 * uw, 2 * uu, 2 * ud))
            self.assertEqual(sum(c.known_counts.windows for c in row.conditions), 2 * kw)
            self.assertEqual(sum(c.unknown_counts.windows for c in row.conditions), 2 * uw)
            for fold in row.folds:
                s = fold.known_vote_stability
                self.assertEqual(s.unanimous_windows + s.majority_not_unanimous_windows + s.tied_windows, kw)
                self.assertAlmostEqual(sum(vars(s.rates).values()), 1)
                self.assertEqual(tuple(c.condition for c in fold.conditions), ("center", "translation_left"))

    def test_aggregation_thresholds_and_all_window_one_deltas(self):
        for row in self.report.results:
            self.assertEqual(row.mean_roc_auc, fmean(f.analysis.roc_auc for f in row.folds))
            self.assertEqual(row.std_roc_auc, pstdev(f.analysis.roc_auc for f in row.folds))
            self.assertEqual(row.mean_eer, fmean(f.analysis.equal_error_point.eer for f in row.folds))
            thresholds = [f.analysis.calibration.threshold for f in row.folds]
            self.assertAlmostEqual(row.median_development_threshold, fmean(thresholds))
            self.assertEqual(row.absolute_threshold_difference, abs(thresholds[0] - thresholds[1]))
            for name in row.mean_metrics.__dataclass_fields__:
                self.assertAlmostEqual(getattr(row.mean_metrics, name),
                                       fmean(getattr(f.analysis.calibration.metrics, name) for f in row.folds))
                self.assertAlmostEqual(getattr(row.std_metrics, name),
                                       pstdev(getattr(f.analysis.calibration.metrics, name) for f in row.folds))
            base = next(r for r in self.report.results if r.score_family == row.score_family and r.window_size == 1)
            expected = (row.mean_roc_auc - base.mean_roc_auc, row.mean_eer - base.mean_eer,
                        row.mean_metrics.unknown_false_acceptance_rate - base.mean_metrics.unknown_false_acceptance_rate,
                        row.mean_metrics.known_rejection_rate - base.mean_metrics.known_rejection_rate,
                        row.mean_metrics.known_correct_identification_rate - base.mean_metrics.known_correct_identification_rate,
                        row.mean_metrics.open_set_exact_accuracy - base.mean_metrics.open_set_exact_accuracy)
            self.assertEqual(tuple(vars(row.vs_window1).values()), expected)

    def test_global_threshold_condition_metrics_and_far_points_match_observations(self):
        fold = create_session_folds(self.known)[0]
        inference = SklearnTemporalOpenSetRunner().infer_fold(*fold.fit.arrays(), *fold.validation.arrays(), self.unknown.X, 3)
        labels = {person: index for index, person in enumerate(self.known.label_to_person)}
        for row in self.report.results:
            kp, up = build_windows(fold.validation.metadata, row.window_size), build_windows(self.unknown.samples, row.window_size)
            kw = aggregate_windows(kp, inference.known.predicted, inference.known.decisions, inference.known.confidence)
            uw = aggregate_windows(up, inference.unknown.predicted, inference.unknown.decisions, inference.unknown.confidence)
            index = TEMPORAL_SCORES.index(row.score_family)
            ks, us = kw.confidence[index][1], uw.confidence[index][1]
            correct = kw.predicted == np.array([labels[w.person_id] for w in kp.windows])
            result = row.folds[0]
            self.assertEqual(result.at_development_threshold,
                             evaluate_threshold(ks, correct, us, row.median_development_threshold, allow_negative=True))
            for condition in result.conditions:
                kmask, umask = np.array(kw.conditions) == condition.condition, np.array(uw.conditions) == condition.condition
                threshold = result.analysis.calibration.threshold
                self.assertAlmostEqual(condition.known_correct_identification_rate, ((ks[kmask] >= threshold) & correct[kmask]).mean())
                self.assertAlmostEqual(condition.known_rejection_rate, (ks[kmask] < threshold).mean())
                self.assertAlmostEqual(condition.unknown_rejection_rate, (us[umask] < threshold).mean())
            self.assertEqual([p.target_far for p in result.analysis.far_operating_points], [.2, .1, .05])
            for point in result.analysis.far_operating_points:
                if point.metrics:
                    self.assertLessEqual(point.metrics.unknown_false_acceptance_rate, point.target_far)
                    self.assertEqual(point.metrics, evaluate_threshold(ks, correct, us, point.threshold, allow_negative=True))

    def test_only_two_fits_and_all_fitted_steps_use_known_fit_only(self):
        models, fit_calls, centroid_calls = [], [], []
        original_centroid_fit = PredictedClassCentroidScorer.fit

        def build(*args):
            model = build_reduction_pipeline(*args)
            original = model.fit

            def fit(X, y):
                fit_calls.append((X.copy(), y.copy()))
                return original(X, y)

            model.fit = fit
            models.append(model)
            return model

        def fit_centroid(scorer, X, y, count):
            centroid_calls.append((X.copy(), y.copy()))
            return original_centroid_fit(scorer, X, y, count)

        with patch("face_recognition.infrastructure.classifiers.sklearn_unknown_rejection.build_reduction_pipeline", build), \
             patch.object(PredictedClassCentroidScorer, "fit", fit_centroid):
            TemporalOpenSetEvaluator(SklearnTemporalOpenSetRunner()).evaluate(self.known, self.unknown)
        self.assertEqual((len(models), len(fit_calls), len(centroid_calls)), (2, 2, 2))
        for fold, model, (fit_X, fit_y), (Z, zy) in zip(create_session_folds(self.known), models, fit_calls, centroid_calls):
            X, y = fold.fit.arrays()
            self.assertTrue(np.array_equal(fit_X, X) and np.array_equal(fit_y, y))
            self.assertTrue(np.allclose(Z, model[:-1].transform(X)) and np.array_equal(zy, y))
            selector = next(step for _, step in model.steps if hasattr(step, "get_support"))
            scaler = next(step for _, step in model.steps if hasattr(step, "mean_"))
            self.assertEqual(selector.k, 448)
            self.assertTrue(np.allclose(scaler.mean_, selector.transform(X).mean(axis=0, dtype=np.float64)))
            classifier = model.named_steps["classifier"]
            self.assertEqual((classifier.C, classifier.gamma, classifier.kernel, classifier.probability), (4., .001, "rbf", False))

    def test_changed_validation_and_unknown_cannot_change_fitted_parameters(self):
        fold = create_session_folds(self.known)[0]
        models = []

        def build(*args):
            model = build_reduction_pipeline(*args)
            models.append(model)
            return model

        with patch("face_recognition.infrastructure.classifiers.sklearn_unknown_rejection.build_reduction_pipeline", build):
            runner = SklearnTemporalOpenSetRunner()
            runner.infer_fold(*fold.fit.arrays(), *fold.validation.arrays(), self.unknown.X, 3)
            runner.infer_fold(*fold.fit.arrays(), fold.validation.X + 100, fold.validation.y, self.unknown.X - 100, 3)
        self.assertTrue(np.array_equal(models[0][0].get_support(), models[1][0].get_support()))
        self.assertTrue(np.array_equal(models[0][1].mean_, models[1][1].mean_))
        self.assertTrue(np.array_equal(models[0][-1].support_vectors_, models[1][-1].support_vectors_))

    def test_protected_or_nonfinite_rows_rejected_before_inference(self):
        for kind in ("known_test", "unknown_holdout", "nan"):
            known, unknown = load_rejection_fixture(self.processed, self.source, self.partition)
            if kind == "known_test":
                held = next(r for r in self.source.samples if r.partition is IdentityPartition.KNOWN_TEST)
                object.__setattr__(known, "samples", (held, *known.samples[1:]))
            elif kind == "unknown_holdout":
                held = next(r for r in self.source.samples if r.person_id in self.partition.specification.holdout_ids)
                object.__setattr__(unknown, "samples", (held, *unknown.samples[1:]))
            else:
                unknown.X.setflags(write=True)
                unknown.X[0, 0] = np.nan
            runner = SklearnTemporalOpenSetRunner()
            with patch.object(runner, "infer_fold") as infer:
                with self.assertRaises(UnknownRejectionError):
                    TemporalOpenSetEvaluator(runner).evaluate(known, unknown)
                infer.assert_not_called()

    def test_ranking_exact_priority_order_and_final_tie(self):
        original = self.report.results[0]

        def row(size=7, family="svm_margin", auc=.6, eer=.4, far=.3, correct=.4, accuracy=.5, frr=.5):
            return replace(original, window_size=size, score_family=family, mean_roc_auc=auc, mean_eer=eer,
                           mean_metrics=replace(original.mean_metrics, unknown_false_acceptance_rate=far,
                                                known_correct_identification_rate=correct, open_set_exact_accuracy=accuracy,
                                                known_rejection_rate=frr))

        for better in (row(auc=.7), row(eer=.3), row(far=.2), row(correct=.5), row(accuracy=.6), row(frr=.4)):
            worse = row(size=1, family="centroid_distance")
            self.assertEqual(rank_temporal_results((worse, better))[0], better.identifier)
        self.assertEqual(rank_temporal_results((row(), row(size=3)))[0], row(size=3).identifier)
        self.assertEqual(rank_temporal_results((row(), row(family="centroid_distance")))[0], row(family="centroid_distance").identifier)
        self.assertEqual(rank_temporal_results(tuple(reversed(self.report.results))), self.report.ranking)

    def test_no_complete_windows_fail_explicitly(self):
        with TemporaryDirectory(prefix="synthetic_short_windows_") as directory:
            processed, source, partition, _, _ = rejection_fixture(Path(directory))
            known, unknown = load_rejection_fixture(processed, source, partition)
            with self.assertRaisesRegex(UnknownRejectionError, "Window 5 has no complete"):
                TemporalOpenSetEvaluator(SklearnTemporalOpenSetRunner()).evaluate(known, unknown)

    def test_runtime_observations_repr_hides_arrays(self):
        fold = create_session_folds(self.known)[0]
        observations = SklearnTemporalOpenSetRunner().infer_fold(*fold.fit.arrays(), *fold.validation.arrays(), self.unknown.X, 3)
        self.assertNotIn("array(", repr(observations))

    def test_known_window_metric_arithmetic_auc_eer_and_threshold_ties(self):
        # Use an independent single-group plan: four complete three-frame windows.
        plan = build_windows(references(12), 3)
        known_values = np.repeat([4., 3., 2., 1.], 3)
        unknown_values = np.repeat([3.5, 2.5, 1.5, .5], 3)
        known = aggregate_windows(plan, np.repeat([0, 1, 0, 1], 3), np.ones((12, 2)), (("svm_margin", known_values),))
        unknown = aggregate_windows(plan, np.zeros(12, dtype=int), np.ones((12, 2)), (("svm_margin", unknown_values),))
        ks, us = known.confidence[0][1], unknown.confidence[0][1]
        correct = known.predicted == 0
        metrics = evaluate_threshold(ks, correct, us, 2.5, allow_negative=True)
        self.assertEqual(metrics.unknown_false_acceptance_rate, .5)
        self.assertEqual(metrics.known_rejection_rate, .5)
        self.assertEqual(metrics.unknown_rejection_rate, .5)
        self.assertEqual(metrics.known_acceptance_rate, .5)
        self.assertEqual(metrics.known_correct_identification_rate, .25)
        self.assertEqual(metrics.known_misidentification_rate, .25)
        self.assertEqual(metrics.open_set_exact_accuracy, .375)
        self.assertEqual(metrics.binary_balanced_accuracy, .5)
        self.assertEqual(metrics.binary_macro_f1, .5)
        analysis = analyze_scores(ks, correct, us)
        self.assertEqual(analysis.roc_auc, .625)
        self.assertEqual(analysis.equal_error_point.eer, .5)
        self.assertEqual(analysis.equal_error_point.threshold, 2.5)
        self.assertEqual(analysis.calibration.metrics.binary_balanced_accuracy, .625)
        self.assertEqual(analysis.calibration.metrics.unknown_false_acceptance_rate, 0)
        self.assertEqual(analysis.calibration.threshold, 4.)

    def test_448_model_does_not_require_experimental_512_feature_minimum(self):
        known = replace(self.known, X=self.known.X[:, :450])
        unknown = replace(self.unknown, X=self.unknown.X[:, :450])
        report = TemporalOpenSetEvaluator(SklearnTemporalOpenSetRunner()).evaluate(known, unknown)
        self.assertEqual(report.original_feature_count, 450)
        self.assertEqual(report.model.k, 448)
