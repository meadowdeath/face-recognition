from dataclasses import replace
from pathlib import Path
from hashlib import sha256
from tempfile import TemporaryDirectory
import unittest
import warnings
from unittest.mock import Mock, patch

import numpy as np

from face_recognition.application import evaluate_v3_novelty as evaluator_module
from face_recognition.application.evaluate_v3_novelty import V3NoveltyEvaluator
from face_recognition.application.rejection_metrics import calibrate_threshold
from face_recognition.application.v3_novelty_calibration import CrossFittedWindowPool, candidate_ranking, evaluate_pair
from face_recognition.application.v3_novelty_data import FrameScores, ScoredPopulation, V3Population, WindowBatch
from face_recognition.domain.models.rejection import RejectionError
from face_recognition.domain.models.temporal import WindowCounts
from face_recognition.domain.models.v3_novelty_evaluation import CANDIDATES
from face_recognition.infrastructure.persistence.v3_novelty_report import v3_novelty_report_to_json
from face_recognition.infrastructure.classifiers.sklearn_v3_novelty import SklearnV3NoveltyRunner
from tests.v3_novelty_fakes import RecordingNoveltyRunner, novelty_data


class V3NoveltyEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory(prefix="synthetic_v3_evaluation_")
        cls.data, _, _ = novelty_data(Path(cls.temporary.name))
        cls.runner = RecordingNoveltyRunner()
        def calibration(*args, **kwargs):
            cls.runner.events.append(("calibrate", len(args[0]), len(args[2])))
            return calibrate_threshold(*args, **kwargs)
        with patch.object(evaluator_module, "calibrate_threshold", calibration):
            cls.report = V3NoveltyEvaluator(cls.runner).evaluate(cls.data, "a" * 64)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_all_candidates_splits_outer_metrics_and_discard_counts(self):
        self.assertEqual(len(self.report.outer_results), 4)
        for row in self.report.outer_results:
            self.assertEqual(len(row.inner_results), 18 * 4)
            self.assertEqual(tuple(s.method for s in row.inner_summaries), CANDIDATES)
            self.assertEqual(row.selected_method, "svm_top1")
            self.assertEqual(row.operational_metrics.open_set_exact_accuracy, 1.)
            self.assertEqual(row.closed_set_known_frames.accuracy, 7 / 8)
            self.assertEqual(row.known_test_counts, WindowCounts(208, 182, 26, 26))
            self.assertEqual(row.unknown_test_counts, WindowCounts(416, 364, 52, 52))
        self.assertEqual(self.report.mean_closed_set.accuracy, 7 / 8)
        self.assertEqual(self.report.std_closed_set.accuracy, 0.)
        self.assertEqual(self.report.mean_roc_auc, 1.)
        self.assertEqual(self.report.std_roc_auc, 0.)
        self.assertEqual(dict(self.report.selection_frequency), {name: 4 if name == "svm_top1" else 0 for name in CANDIDATES})

    def test_cache_fits_known_only_and_test_inference_occurs_once_after_calibration(self):
        events = self.runner.events
        indexed_fits = [(i, event) for i, event in enumerate(events) if event[0] == "fit"]
        fits = [event for _, event in indexed_fits]
        self.assertEqual(len(fits), 4 * (3 + 1))
        self.assertEqual(sum(e[0] == "calibrate" for e in events), 4 * (18 * 4 + 1))
        for index, outer in enumerate(self.data.protocol.outer_folds):
            group = fits[index * 4:index * 4 + 4]
            self.assertEqual({event[1] for event in group[:3]}, {(s,) for s in outer.known_development_sessions})
            self.assertEqual(group[-1][1], outer.known_development_sessions)
            self.assertEqual(group[-1][3], 3 * group[0][3])
            self.assertTrue(all(e[2] == self.data.protocol.specification.known_ids for e in group))
            self.assertEqual((self.report.outer_results[index].cached_inner_fit_populations,
                              self.report.outer_results[index].cache_reuses), (3, 15))
            final_fit_index = indexed_fits[index * 4 + 3][0]
            self.assertEqual(events[final_fit_index - 1][0], "calibrate")
            later = events[final_fit_index + 1:]
            known_tests = [e for e in later if e[0] == "infer" and e[1] == outer.known_development_sessions
                           and e[2] == (outer.known_test_session,)]
            unknown_tests = [e for e in later if e[0] == "infer" and e[1] == outer.known_development_sessions
                             and e[3] == outer.unknown_test_ids]
            self.assertEqual((len(known_tests), len(unknown_tests)), (1, 1))
            # Every inner model scores only other development sessions or six
            # development UNKNOWN identities, never this outer test population.
            start = indexed_fits[index * 4][0]
            for e in events[start:final_fit_index]:
                if e[0] == "infer":
                    if e[3] == self.data.protocol.specification.known_ids:
                        self.assertTrue(set(e[2]) <= set(outer.known_development_sessions))
                        self.assertFalse(set(e[1]) & set(e[2]))
                    else:
                        self.assertEqual(e[3], outer.unknown_development_ids)

    def test_pool_counts_remove_rotation_duplicates_and_equal_weight_distinct_models(self):
        for result in self.report.outer_results:
            k, u = result.calibration.known_pool, result.calibration.unknown_pool
            self.assertEqual((k.unique_physical_windows, k.distinct_model_window_observations, k.raw_window_observations), (78, 156, 468))
            self.assertEqual((u.unique_physical_windows, u.distinct_model_window_observations, u.raw_window_observations), (156, 468, 1872))
        pool = CrossFittedWindowPool()
        def batch(value):
            return WindowBatch((("synthetic", "visit", "center", tuple(range(7))),), np.array([value]), np.array([True]), WindowCounts(7, 7, 0, 1))
        for _ in range(10):
            pool.add("fit_a", batch(1.))
        pool.add("fit_b", batch(9.))
        scores, counts = pool.finish(2)
        self.assertEqual(scores.item(), 5.)  # Not the rotation-weighted median 1.
        self.assertEqual(counts.raw_window_observations, 11)
        with self.assertRaises(RejectionError):
            pool.add("fit_a", batch(2.))
        with self.assertRaises(RejectionError):
            pool.finish(3)

    def test_every_threshold_uses_only_its_declared_calibration_roles(self):
        calls, batches, selected_thresholds = [], [], []
        original_batch = ScoredPopulation.batch
        def batch(self, method, **roles):
            result = original_batch(self, method, **roles)
            batches.append((method, roles, result))
            return result
        def calibrate(known, correct, unknown, **kwargs):
            outer_index, inner_call = divmod(len(calls), 73)
            outer = self.data.protocol.outer_folds[outer_index]
            if inner_call < 72:
                split = outer.inner_splits[inner_call // 4]
                self.assertEqual(batches[-2][1], {"sessions": (split.known_calibration_session,)})
                self.assertEqual(batches[-1][1], {"identities": split.unknown_calibration_ids})
                np.testing.assert_array_equal(known, batches[-2][2].scores)
                np.testing.assert_array_equal(unknown, batches[-1][2].scores)
            result = calibrate_threshold(known, correct, unknown, **kwargs)
            calls.append(result.threshold)
            selected_thresholds.append(result.threshold)
            return result
        original_pair = evaluate_pair
        def validation(known, unknown, threshold):
            self.assertEqual(threshold, selected_thresholds[-1])
            return original_pair(known, unknown, threshold)
        with patch.object(ScoredPopulation, "batch", batch), patch.object(evaluator_module, "calibrate_threshold", calibrate), \
                patch.object(evaluator_module, "evaluate_pair", validation):
            V3NoveltyEvaluator(RecordingNoveltyRunner()).evaluate(self.data, "a" * 64)
        self.assertEqual(len(calls), 292)

    def test_exact_threshold_ties_and_validation_cannot_recalibrate(self):
        known = WindowBatch((), np.array([.9, .1, .8]), np.array([True, True, False]), WindowCounts(21, 21, 0, 3))
        unknown = WindowBatch((), np.array([.2, .8]), np.zeros(2, dtype=bool), WindowCounts(14, 14, 0, 2))
        with patch("face_recognition.application.rejection_metrics.calibrate_threshold", side_effect=AssertionError("Validation calibration forbidden")):
            metrics, diagnostics = evaluate_pair(known, unknown, .75)
        self.assertEqual(metrics.open_set_exact_accuracy, 2 / 5)
        self.assertEqual(metrics.known_correct_identification_rate, 1 / 3)
        self.assertEqual(metrics.known_misidentification_rate, 1 / 3)
        self.assertEqual(metrics.known_rejection_rate, 1 / 3)
        self.assertEqual(metrics.unknown_false_acceptance_rate, .5)
        self.assertNotEqual(diagnostics.equal_error_point.threshold, .75)
        threshold = calibrate_threshold(np.array([.8, .9]), np.ones(2, dtype=bool), np.array([.1, .2])).threshold
        self.assertEqual(threshold, .8)  # Equal decisions: higher numeric threshold.

    def test_candidate_ranking_uses_declared_criteria_in_order(self):
        originals = self.report.outer_results[0].inner_summaries
        base = tuple(replace(s, mean_metrics=replace(s.mean_metrics, binary_balanced_accuracy=.7,
                     unknown_false_acceptance_rate=.2, known_rejection_rate=.3), mean_roc_auc=.8) for s in originals)
        self.assertEqual(candidate_ranking(base), CANDIDATES)
        for field, value in (("binary_balanced_accuracy", .8), ("unknown_false_acceptance_rate", .1), ("known_rejection_rate", .2)):
            rows = list(base)
            rows[-1] = replace(rows[-1], mean_metrics=replace(rows[-1].mean_metrics, **{field: value}))
            self.assertEqual(candidate_ranking(tuple(rows))[0], "mahalanobis_distance")
        rows = (*base[:-1], replace(base[-1], mean_roc_auc=.9))
        self.assertEqual(candidate_ranking(rows)[0], "mahalanobis_distance")

    def test_report_json_is_byte_identical_without_timing_or_per_sample_values(self):
        second = V3NoveltyEvaluator(RecordingNoveltyRunner()).evaluate(self.data, "a" * 64)
        self.assertEqual(v3_novelty_report_to_json(second), v3_novelty_report_to_json(self.report))

    def test_real_sklearn_nested_execution_and_json_are_deterministic(self):
        # Marker columns intentionally encode constant class/session metadata;
        # their ANOVA constant-feature warnings are synthetic-fixture artifacts.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            first = V3NoveltyEvaluator(SklearnV3NoveltyRunner()).evaluate(self.data, "a" * 64)
            second = V3NoveltyEvaluator(SklearnV3NoveltyRunner()).evaluate(self.data, "a" * 64)
        self.assertEqual(v3_novelty_report_to_json(first), v3_novelty_report_to_json(second))
        self.assertEqual(len(first.outer_results), 4)

    def test_unsafe_transport_shift_is_diagnostic_and_never_changes_threshold(self):
        result = V3NoveltyEvaluator(RecordingNoveltyRunner(final_shift=100.)).evaluate(self.data, "a" * 64)
        self.assertEqual([r.threshold for r in result.outer_results], [r.threshold for r in self.report.outer_results])
        self.assertTrue(all(r.transport.warning for r in result.outer_results))
        self.assertTrue(all(r.operational_metrics.unknown_false_acceptance_rate == 1. for r in result.outer_results))

    def test_changing_outer_test_values_cannot_change_that_folds_selection_or_threshold(self):
        outer = self.data.protocol.outer_folds[0]
        X = self.data.X.copy()
        for i, sample in enumerate(self.data.samples):
            if sample.person_id in outer.unknown_test_ids:
                X[i, 0] = 0.  # Fake runner now regards these unseen UNKNOWNs as known.
            elif sample.session_id == outer.known_test_session:
                X[i, 2] = 0.  # Remove the fake trailing-frame identity error.
        changed = replace(self.data, X=X, payload_sha256=sha256(X.tobytes()).hexdigest())
        result = V3NoveltyEvaluator(RecordingNoveltyRunner()).evaluate(changed, "a" * 64)
        before, after = self.report.outer_results[0], result.outer_results[0]
        self.assertEqual(before.inner_results, after.inner_results)
        self.assertEqual((before.selected_method, before.threshold, before.calibration),
                         (after.selected_method, after.threshold, after.calibration))
        self.assertNotEqual(before.operational_metrics, after.operational_metrics)

    def test_corrupted_session_or_unknown_role_is_rejected_before_fitting(self):
        original = self.data.protocol
        outer = original.outer_folds[0]
        for split in (replace(outer.inner_splits[0], known_fit_sessions=(outer.known_test_session,)),
                      replace(outer.inner_splits[0], unknown_calibration_ids=outer.unknown_test_ids)):
            changed = replace(original, outer_folds=(replace(outer, inner_splits=(split, *outer.inner_splits[1:])), *original.outer_folds[1:]))
            runner = Mock()
            try:
                object.__setattr__(self.data, "protocol", changed)
                with self.assertRaises(ValueError):
                    V3NoveltyEvaluator(runner).evaluate(self.data, "a" * 64)
                runner.fit.assert_not_called()
            finally:
                object.__setattr__(self.data, "protocol", original)

    def test_window_seven_median_and_existing_tie_breaking(self):
        source = self.data.known((self.data.protocol.specification.known_sessions[0],))
        population = V3Population(source.X[:8], source.y[:8], source.samples[:8])
        predictions = np.array([0, 0, 0, 1, 1, 1, 2, 1])
        decisions = np.zeros((8, 3)); decisions[:, 1] = 1.
        scores = np.array([-100., -90., -80., 1., 2., 3., 4., 1000.])
        batch = ScoredPopulation(population, FrameScores(predictions, decisions, (("svm_top1", scores),))).batch("svm_top1")
        self.assertEqual(batch.scores.item(), 1.)
        self.assertFalse(batch.correct.item())  # Summed scores resolve tied votes to class 1.
        self.assertEqual(batch.counts, WindowCounts(8, 7, 1, 1))
        decisions[:] = 0.
        batch = ScoredPopulation(population, FrameScores(predictions, decisions, (("svm_top1", scores),))).batch("svm_top1")
        self.assertTrue(batch.correct.item())  # Exact final tie resolves by class order.
