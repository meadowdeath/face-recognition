from dataclasses import replace
from pathlib import Path
from statistics import fmean, pstdev
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from face_recognition.application.evaluate_distance_novelty import DistanceNoveltyEvaluator
from face_recognition.application.evaluate_temporal_distance_novelty import TemporalDistanceNoveltyEvaluator, rank_temporal_distance_results
from face_recognition.application.rejection_metrics import evaluate_threshold
from face_recognition.application.rejection_score_analysis import analyze_scores
from face_recognition.application.temporal_windows import aggregate_windows, build_windows, vote_identity
from face_recognition.application.training_folds import create_session_folds
from face_recognition.application.unknown_rejection_split import UnknownRejectionError
from face_recognition.domain.models.identity_split import IdentityPartition
from face_recognition.domain.models.temporal_distance_novelty import TEMPORAL_DISTANCE_SCORES
from face_recognition.domain.models.temporal_open_set import WINDOW_SIZES
from face_recognition.infrastructure.classifiers.sklearn_distance_novelty import SklearnDistanceNoveltyRunner
from face_recognition.infrastructure.classifiers.sklearn_feature_reduction import build_reduction_pipeline
from face_recognition.infrastructure.classifiers.sklearn_temporal_distance_novelty import SklearnTemporalDistanceNoveltyRunner
from face_recognition.infrastructure.features.known_distances import KnownDistanceScorer
from face_recognition.infrastructure.features.rejection_scores import PredictedClassCentroidScorer
from tests.unknown_rejection_fakes import rejection_fixture, load_rejection_fixture
from tests.unit.test_temporal_windows import references


class TemporalDistanceNoveltyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory(prefix="synthetic_temporal_distance_")
        cls.processed, cls.source, cls.partition, _, _ = rejection_fixture(Path(cls.temporary.name), samples=8)
        cls.known, cls.unknown = load_rejection_fixture(cls.processed, cls.source, cls.partition)
        cls.report = TemporalDistanceNoveltyEvaluator(SklearnTemporalDistanceNoveltyRunner()).evaluate(cls.known, cls.unknown)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_exact_two_methods_four_window_sizes_and_two_folds(self):
        self.assertEqual([(row.score_family, row.window_size) for row in self.report.results],
                         [(name, size) for name in ("nearest_1", "mean_5nn") for size in (1, 3, 5, 7)])
        self.assertTrue(all(len(row.folds) == 2 for row in self.report.results))
        self.assertEqual((self.report.model.k, self.report.model.C, self.report.model.gamma, self.report.model.kernel), (448, 4., .001, "rbf"))
        self.assertFalse(self.report.model.probability)

    def test_window_one_exactly_reproduces_frame_level_distance_reference(self):
        frame = DistanceNoveltyEvaluator(SklearnDistanceNoveltyRunner()).evaluate(self.known, self.unknown)
        for row in self.report.results:
            if row.window_size != 1:
                continue
            original = next(r for r in frame.novelty_results if r.score_family == row.score_family)
            self.assertEqual(row.mean_metrics, original.mean_metrics)
            self.assertEqual(row.mean_roc_auc, original.mean_roc_auc)
            self.assertEqual(row.mean_eer, original.mean_eer)
            for temporal, previous in zip(row.folds, original.folds):
                self.assertEqual(temporal.analysis, previous.analysis)
                self.assertEqual(temporal.frame_closed_set_metrics, previous.closed_set_metrics)
                self.assertEqual(temporal.known_vote_stability.rates.unanimous_rate, 1.)

    def test_sixteen_independent_fold_method_window_calibrations(self):
        calls = []

        def analyze(known, correct, unknown):
            calls.append((len(known), len(correct), len(unknown)))
            return analyze_scores(known, correct, unknown)

        with patch("face_recognition.application.evaluate_temporal_open_set.analyze_scores", analyze):
            TemporalDistanceNoveltyEvaluator(SklearnTemporalDistanceNoveltyRunner()).evaluate(self.known, self.unknown)
        self.assertEqual(calls, ([(48, 48, 32)] * 2 + [(12, 12, 8)] * 2 + [(6, 6, 4)] * 4) * 2)

    def test_fit_only_known_references_preprocessing_and_only_two_requested_distances(self):
        models, fit_inputs, references_seen = [], [], []
        original = KnownDistanceScorer.fit

        def build(*args):
            model = build_reduction_pipeline(*args)
            fit = model.fit

            def tracked_fit(X, y):
                fit_inputs.append((X.copy(), y.copy()))
                return fit(X, y)

            model.fit = tracked_fit
            models.append(model)
            return model

        def fit_reference(scorer, X, y, count):
            original(scorer, X, y, count)
            references_seen.append((scorer._reference.copy(), scorer._labels.copy(), scorer._scores))
            return scorer

        with patch("face_recognition.infrastructure.classifiers.sklearn_unknown_rejection.build_reduction_pipeline", build), \
             patch.object(KnownDistanceScorer, "fit", fit_reference), \
             patch.object(PredictedClassCentroidScorer, "fit", side_effect=AssertionError("No centroid experiment here")):
            TemporalDistanceNoveltyEvaluator(SklearnTemporalDistanceNoveltyRunner()).evaluate(self.known, self.unknown)
        self.assertEqual((len(models), len(fit_inputs), len(references_seen)), (2, 2, 2))
        for fold, model, (X, y), (Z, zy, scores) in zip(create_session_folds(self.known), models, fit_inputs, references_seen):
            self.assertEqual(scores, (("nearest_1", 1), ("mean_5nn", 5)))
            self.assertTrue(np.array_equal(X, fold.fit.X) and np.array_equal(y, fold.fit.y))
            self.assertTrue(np.allclose(Z, model[:-1].transform(fold.fit.X)) and np.array_equal(zy, fold.fit.y))
            self.assertEqual(model[0].k, 448)
            self.assertTrue(np.allclose(model[1].mean_, model[0].transform(X).mean(axis=0, dtype=np.float64)))
            self.assertEqual((model[-1].C, model[-1].gamma, model[-1].kernel, model[-1].probability), (4., .001, "rbf", False))

    def test_validation_and_unknown_changes_do_not_change_fit_or_references(self):
        fold = create_session_folds(self.known)[0]
        models, reference_sets = [], []
        original = KnownDistanceScorer.fit

        def fit_reference(scorer, X, y, count):
            original(scorer, X, y, count)
            reference_sets.append((scorer._reference.copy(), scorer._labels.copy()))
            return scorer

        def build(*args):
            model = build_reduction_pipeline(*args)
            models.append(model)
            return model

        with patch.object(KnownDistanceScorer, "fit", fit_reference), \
             patch("face_recognition.infrastructure.classifiers.sklearn_unknown_rejection.build_reduction_pipeline", build):
            runner = SklearnTemporalDistanceNoveltyRunner()
            runner.infer_fold(*fold.fit.arrays(), *fold.validation.arrays(), self.unknown.X, 3)
            runner.infer_fold(*fold.fit.arrays(), fold.validation.X + 100, fold.validation.y, self.unknown.X - 100, 3)
        self.assertTrue(np.array_equal(reference_sets[0][0], reference_sets[1][0]))
        self.assertTrue(np.array_equal(reference_sets[0][1], reference_sets[1][1]))
        self.assertTrue(np.array_equal(models[0][0].get_support(), models[1][0].get_support()))
        self.assertTrue(np.array_equal(models[0][1].mean_, models[1][1].mean_))
        self.assertTrue(np.array_equal(models[0][-1].support_vectors_, models[1][-1].support_vectors_))

    def test_neighbor_identity_does_not_replace_svm_identity(self):
        fold = create_session_folds(self.known)[0]
        runner = SklearnTemporalDistanceNoveltyRunner()
        original = runner.infer_fold(*fold.fit.arrays(), *fold.validation.arrays(), self.unknown.X, 3)
        score = KnownDistanceScorer.score

        def altered(scorer, X):
            observation = score(scorer, X)
            return replace(observation, nearest_labels=(observation.nearest_labels + 1) % 3)

        with patch.object(KnownDistanceScorer, "score", altered):
            other = runner.infer_fold(*fold.fit.arrays(), *fold.validation.arrays(), self.unknown.X, 3)
        self.assertTrue(np.array_equal(original.known.predicted, other.known.predicted))
        self.assertTrue(np.array_equal(original.known.decisions, other.known.decisions))
        self.assertEqual(original.closed_set_metrics, other.closed_set_metrics)
        self.assertNotIn("array(", repr(other))

    def test_used_discarded_counts_and_no_window_boundary_crossing(self):
        fold = create_session_folds(self.known)[0]
        expected = {1: (48, 48, 0, 32, 32, 0), 3: (12, 36, 12, 8, 24, 8),
                    5: (6, 30, 18, 4, 20, 12), 7: (6, 42, 6, 4, 28, 4)}
        for row in self.report.results:
            kw, ku, kd, uw, uu, ud = expected[row.window_size]
            self.assertEqual((row.total_known_counts.windows, row.total_known_counts.used_samples, row.total_known_counts.discarded_samples),
                             (2 * kw, 2 * ku, 2 * kd))
            self.assertEqual((row.total_unknown_counts.windows, row.total_unknown_counts.used_samples, row.total_unknown_counts.discarded_samples),
                             (2 * uw, 2 * uu, 2 * ud))
            plan = build_windows(tuple(reversed(fold.validation.metadata)), row.window_size)
            metadata = tuple(reversed(fold.validation.metadata))
            indices = []
            for window in plan.windows:
                self.assertEqual({(metadata[i].person_id, metadata[i].session_id, metadata[i].condition) for i in window.indices},
                                 {(window.person_id, window.session_id, window.condition)})
                ordered = [metadata[i].sample_index for i in window.indices]
                self.assertEqual(ordered, list(range(ordered[0], ordered[0] + row.window_size)))
                indices.extend(window.indices)
            self.assertEqual(len(indices), len(set(indices)))

    def test_twenty_sample_efficiency_is_reported_without_favoring_five(self):
        samples = references(20)
        for size, windows, used, discarded in ((1, 20, 20, 0), (3, 6, 18, 2), (5, 4, 20, 0), (7, 2, 14, 6)):
            counts = build_windows(samples, size).counts
            self.assertEqual((counts.windows, counts.used_samples, counts.discarded_samples), (windows, used, discarded))

    def test_aggregate_metrics_and_same_method_frame_deltas(self):
        for row in self.report.results:
            self.assertEqual(row.mean_roc_auc, fmean(f.analysis.roc_auc for f in row.folds))
            self.assertEqual(row.std_roc_auc, pstdev(f.analysis.roc_auc for f in row.folds))
            self.assertEqual(row.mean_eer, fmean(f.analysis.equal_error_point.eer for f in row.folds))
            self.assertEqual(row.std_eer, pstdev(f.analysis.equal_error_point.eer for f in row.folds))
            thresholds = [f.analysis.calibration.threshold for f in row.folds]
            self.assertAlmostEqual(row.median_development_threshold, fmean(thresholds))
            self.assertEqual(row.absolute_threshold_difference, abs(thresholds[0] - thresholds[1]))
            for name in row.mean_metrics.__dataclass_fields__:
                self.assertAlmostEqual(getattr(row.mean_metrics, name), fmean(getattr(f.analysis.calibration.metrics, name) for f in row.folds))
                self.assertAlmostEqual(getattr(row.std_metrics, name), pstdev(getattr(f.analysis.calibration.metrics, name) for f in row.folds))
            base = next(r for r in self.report.results if r.window_size == 1 and r.score_family == row.score_family)
            self.assertEqual(tuple(vars(row.vs_window1).values()), (
                row.mean_roc_auc - base.mean_roc_auc, row.mean_eer - base.mean_eer,
                row.mean_metrics.unknown_false_acceptance_rate - base.mean_metrics.unknown_false_acceptance_rate,
                row.mean_metrics.known_rejection_rate - base.mean_metrics.known_rejection_rate,
                row.mean_metrics.known_correct_identification_rate - base.mean_metrics.known_correct_identification_rate,
                row.mean_metrics.open_set_exact_accuracy - base.mean_metrics.open_set_exact_accuracy,
            ))

    def test_condition_and_far_metrics_use_global_fold_thresholds(self):
        fold = create_session_folds(self.known)[0]
        inference = SklearnTemporalDistanceNoveltyRunner().infer_fold(*fold.fit.arrays(), *fold.validation.arrays(), self.unknown.X, 3)
        labels = {person: index for index, person in enumerate(self.known.label_to_person)}
        for row in self.report.results:
            kp, up = build_windows(fold.validation.metadata, row.window_size), build_windows(self.unknown.samples, row.window_size)
            kw = aggregate_windows(kp, inference.known.predicted, inference.known.decisions, inference.known.confidence)
            uw = aggregate_windows(up, inference.unknown.predicted, inference.unknown.decisions, inference.unknown.confidence)
            index = TEMPORAL_DISTANCE_SCORES.index(row.score_family)
            ks, us = kw.confidence[index][1], uw.confidence[index][1]
            correct = kw.predicted == np.array([labels[w.person_id] for w in kp.windows])
            result = row.folds[0]
            self.assertEqual(result.at_development_threshold,
                             evaluate_threshold(ks, correct, us, row.median_development_threshold, allow_negative=True))
            for c in result.conditions:
                kmask, umask = np.array(kw.conditions) == c.condition, np.array(uw.conditions) == c.condition
                threshold = result.analysis.calibration.threshold
                self.assertEqual(c.known_counts.windows, int(kmask.sum()))
                self.assertEqual(c.unknown_counts.windows, int(umask.sum()))
                self.assertAlmostEqual(c.known_correct_identification_rate, ((ks[kmask] >= threshold) & correct[kmask]).mean())
                self.assertAlmostEqual(c.known_rejection_rate, (ks[kmask] < threshold).mean())
                self.assertAlmostEqual(c.unknown_rejection_rate, (us[umask] < threshold).mean())
            self.assertEqual([p.target_far for p in result.analysis.far_operating_points], [.2, .1, .05])
            for point in result.analysis.far_operating_points:
                if point.metrics:
                    self.assertLessEqual(point.metrics.unknown_false_acceptance_rate, point.target_far)
                    self.assertEqual(point.metrics, evaluate_threshold(ks, correct, us, point.threshold, allow_negative=True))
            self.assertEqual(sum(c.known_counts.windows for c in row.conditions), row.total_known_counts.windows)

    def test_median_novelty_score_and_all_vote_categories_and_ties(self):
        plan = build_windows(references(9), 3)
        predicted, decisions = np.array([0, 0, 0, 0, 0, 1, 0, 1, 2]), np.ones((9, 3))
        scores = (("nearest_1", np.tile([-9., -1., -2.], 3)), ("mean_5nn", np.tile([-10., -3., -4.], 3)))
        result = aggregate_windows(plan, predicted, decisions, scores)
        self.assertTrue(np.array_equal(result.predicted, np.zeros(3)))
        self.assertTrue(np.array_equal(result.confidence[0][1], np.full(3, -2.)))
        self.assertTrue(np.array_equal(result.confidence[1][1], np.full(3, -4.)))
        self.assertEqual(tuple(vars(result.vote_stability.rates).values()), (1 / 3, 1 / 3, 1 / 3))
        self.assertEqual(vote_identity(np.array([0, 1, 2]), np.array([[5., 0, 0], [0, 4, 0], [0, 10, 3]])), (1, "tie"))
        self.assertEqual(vote_identity(np.array([2, 1, 0]), np.ones((3, 3))), (0, "tie"))
        for invalid in (np.nan, np.inf, -np.inf):
            values = scores[0][1].copy()
            values[-1] = invalid
            with self.assertRaises(UnknownRejectionError):
                aggregate_windows(plan, predicted, decisions, (("nearest_1", values),))

    def test_protected_or_nonfinite_rows_fail_before_fitting(self):
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
            runner = SklearnTemporalDistanceNoveltyRunner()
            with patch.object(runner, "infer_fold") as infer:
                with self.assertRaises(UnknownRejectionError):
                    TemporalDistanceNoveltyEvaluator(runner).evaluate(known, unknown)
                infer.assert_not_called()

    def test_ranking_exact_priority_window_then_neighbor_count(self):
        original = self.report.results[0]

        def row(size=7, family="mean_5nn", auc=.7, eer=.3, far=.3, correct=.4, accuracy=.5, frr=.5):
            return replace(original, window_size=size, score_family=family, mean_roc_auc=auc, mean_eer=eer,
                           mean_metrics=replace(original.mean_metrics, unknown_false_acceptance_rate=far,
                                                known_correct_identification_rate=correct, open_set_exact_accuracy=accuracy,
                                                known_rejection_rate=frr))

        for better in (row(auc=.8), row(eer=.2), row(far=.2), row(correct=.5), row(accuracy=.6), row(frr=.4)):
            worse = row(size=1, family="nearest_1")
            self.assertEqual(rank_temporal_distance_results((worse, better))[0], better.identifier)
        self.assertEqual(rank_temporal_distance_results((row(size=5), row(size=3)))[0], row(size=3).identifier)
        self.assertEqual(rank_temporal_distance_results((row(size=5, family="nearest_1"), row(size=3)))[0], row(size=3).identifier)
        self.assertEqual(rank_temporal_distance_results((row(), row(family="nearest_1")))[0], row(family="nearest_1").identifier)
        self.assertEqual(rank_temporal_distance_results(tuple(reversed(self.report.results))), self.report.ranking)

    def test_metric_arithmetic_score_direction_and_impractical_far_points(self):
        known, unknown, correct = np.array([-1., -2., -3., -4.]), np.array([-1.5, -2.5, -3.5, -4.5]), np.array([True, False, True, False])
        m = evaluate_threshold(known, correct, unknown, -2.5, allow_negative=True)
        self.assertEqual((m.unknown_false_acceptance_rate, m.known_rejection_rate, m.known_acceptance_rate, m.unknown_rejection_rate), (.5, .5, .5, .5))
        self.assertEqual((m.known_correct_identification_rate, m.known_misidentification_rate, m.open_set_exact_accuracy), (.25, .25, .375))
        self.assertEqual((m.binary_balanced_accuracy, m.binary_macro_f1), (.5, .5))
        analysis = analyze_scores(known, correct, unknown)
        self.assertEqual(analysis.roc_auc, .625)
        self.assertEqual(analysis.equal_error_point.eer, .5)
        impractical = analyze_scores(np.array([-9., -10.]), np.array([True, True]), np.array([-1., -2.]))
        self.assertTrue(all(p.status == "impractical" for p in impractical.far_operating_points))
