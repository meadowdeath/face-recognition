from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from face_recognition.application.evaluate_final_open_set import FinalOpenSetEvaluator
from face_recognition.application.final_evaluation_data import FinalEvaluationError
from face_recognition.application.rejection_metrics import evaluate_threshold
from face_recognition.application.rejection_score_analysis import analyze_scores, score_separation
from face_recognition.application.temporal_windows import aggregate_windows, build_windows
from face_recognition.domain.models.final_evaluation import FinalClosedSet, IdentityClassMetrics
from face_recognition.domain.models.identity_split import IdentityPartition
from face_recognition.infrastructure.classifiers.sklearn_baselines import classification_measurements
from face_recognition.infrastructure.classifiers.sklearn_final_open_set import SklearnFinalOpenSetRunner
from face_recognition.infrastructure.classifiers.sklearn_temporal_open_set import FrameInference
from face_recognition.infrastructure.persistence.model_repository import model_fingerprints
from tests.final_evaluation_fakes import final_fixture, load_final_fixture


class FinalOpenSetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory(prefix="synthetic_final_evaluation_")
        cls.bundle = final_fixture(Path(cls.temporary.name))
        cls.data, cls.model, cls.context = load_final_fixture(cls.bundle)
        cls.report = FinalOpenSetEvaluator(SklearnFinalOpenSetRunner(cls.model)).evaluate(cls.data, cls.context)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_only_final_populations_counts_frames_windows_and_trailing_remainder(self):
        report = self.report
        self.assertEqual((report.known_test.samples, report.unknown_holdout.samples), (48, 48))
        for counts in (report.known_windows, report.unknown_windows):
            self.assertEqual((counts.windows, counts.used_samples, counts.discarded_samples), (6, 42, 6))
        self.assertEqual(report.closed_set_known_frames.samples, 48)
        self.assertEqual(sum(c.support for c in report.closed_set_known_frames.per_class), 48)
        self.assertEqual(sum(map(sum, report.closed_set_known_frames.confusion_matrix)), 48)
        self.assertEqual(report.closed_set_known_frames.label_order, self.model.parts.class_ids)
        self.assertTrue(all(s.partition is IdentityPartition.KNOWN_TEST for s in self.data.known_samples))
        self.assertTrue(all(s.person_id in self.data.partition.specification.holdout_ids for s in self.data.unknown_samples))

    def test_no_fit_fit_transform_or_threshold_calibration_including_model_loading(self):
        forbidden = ("sklearn.feature_selection.SelectKBest.fit", "sklearn.feature_selection.SelectKBest.fit_transform",
                     "sklearn.preprocessing.StandardScaler.fit", "sklearn.preprocessing.StandardScaler.fit_transform",
                     "sklearn.svm.SVC.fit", "sklearn.pipeline.Pipeline.fit", "sklearn.pipeline.Pipeline.fit_transform",
                     "face_recognition.infrastructure.features.known_distances.KnownDistanceScorer.fit",
                     "face_recognition.application.rejection_metrics.calibrate_threshold",
                     "face_recognition.application.rejection_score_analysis.calibrate_threshold",
                     "face_recognition.application.rejection_score_analysis.analyze_scores")
        with ExitStack() as stack:
            for name in forbidden:
                stack.enter_context(patch(name, side_effect=AssertionError("No fitting or operational calibration allowed")))
            data, model, context = load_final_fixture(self.bundle)
            report = FinalOpenSetEvaluator(SklearnFinalOpenSetRunner(model)).evaluate(data, context)
        self.assertEqual(report.operational_metrics, self.report.operational_metrics)
        self.assertEqual(report.context.threshold, self.model.threshold)

    def test_diagnostics_preserve_existing_eer_definition_without_calibration(self):
        known, unknown = np.array([-1., -2., -3., -4.]), np.array([-1.5, -2.5, -3.5, -4.5])
        original = analyze_scores(known, np.ones(4, dtype=bool), unknown)
        with patch("face_recognition.application.rejection_score_analysis.calibrate_threshold", side_effect=AssertionError("Forbidden")):
            diagnostic = score_separation(known, unknown)
        self.assertEqual(diagnostic.roc_auc, .625)
        self.assertEqual(diagnostic.equal_error_point, original.equal_error_point)
        self.assertTrue(diagnostic.diagnostic_only)

    def fake_runner(self, *, reject_all=False):
        data, context = self.data, self.context
        kp, up = build_windows(data.known_samples, 7), build_windows(data.unknown_samples, 7)
        predicted = data.known_y.copy()
        known_scores = np.full(len(predicted), context.threshold - 1.)
        unknown_scores = np.full(len(data.unknown_samples), context.threshold - 2.)
        if not reject_all:
            for index, window in enumerate(kp.windows):
                rows = list(window.indices)
                if index % 3 != 2:
                    known_scores[rows] = context.threshold + 1.
                if index % 3 == 1:
                    predicted[rows] = (predicted[rows] + 1) % 3
            unknown_scores[list(up.windows[0].indices)] = context.threshold + 2.
        known_decisions = np.eye(3)[predicted]
        unknown_predictions = np.zeros(len(unknown_scores), dtype=np.int32)
        metrics, matrix = classification_measurements(data.known_y, predicted, 3)
        per_class = tuple(IdentityClassMetrics(person, 0., 0., 0., int((data.known_y == i).sum())) for i, person in enumerate(data.class_ids))
        closed = FinalClosedSet(len(predicted), metrics, data.class_ids, matrix, per_class)
        result = SimpleNamespace(known=FrameInference(predicted, known_decisions, (("mean_5nn", known_scores),)),
                                 unknown=FrameInference(unknown_predictions, np.eye(3)[unknown_predictions], (("mean_5nn", unknown_scores),)),
                                 closed_set=closed)
        return SimpleNamespace(frozen_threshold=context.threshold, class_ids=data.class_ids,
                               versions=context.model_versions, infer=lambda *args: result), result

    def test_eer_threshold_never_replaces_persisted_operational_threshold(self):
        runner, _ = self.fake_runner(reject_all=True)
        report = FinalOpenSetEvaluator(runner).evaluate(self.data, self.context)
        self.assertNotEqual(report.temporal_novelty_diagnostics.equal_error_point.threshold, self.context.threshold)
        self.assertEqual(report.temporal_novelty_diagnostics.roc_auc, 1.)
        self.assertEqual(report.temporal_novelty_diagnostics.equal_error_point.eer, 0.)
        self.assertEqual(report.operational_metrics.known_rejection_rate, 1.)
        self.assertEqual(report.operational_metrics.unknown_false_acceptance_rate, 0.)
        self.assertEqual(report.operational_metrics.open_set_exact_accuracy, .5)
        self.assertEqual(report.context.threshold, self.model.threshold)
        with self.assertRaises(FinalEvaluationError):
            FinalOpenSetEvaluator(runner).evaluate(self.data, replace(self.context, threshold=self.context.threshold + 1))

    def test_exact_operational_metric_arithmetic_and_closed_set_frame_accuracy(self):
        runner, _ = self.fake_runner()
        report = FinalOpenSetEvaluator(runner).evaluate(self.data, self.context)
        m = report.operational_metrics
        self.assertEqual((m.known_acceptance_rate, m.known_rejection_rate), (4 / 6, 2 / 6))
        self.assertEqual((m.unknown_rejection_rate, m.unknown_false_acceptance_rate), (5 / 6, 1 / 6))
        self.assertEqual((m.known_correct_identification_rate, m.known_misidentification_rate), (2 / 6, 2 / 6))
        self.assertEqual(m.open_set_exact_accuracy, 7 / 12)
        self.assertEqual(m.binary_balanced_accuracy, .75)
        closed = report.closed_set_known_frames
        self.assertEqual(closed.confusion_matrix, ((9, 7, 0), (0, 16, 0), (7, 0, 9)))
        self.assertEqual(closed.metrics.accuracy, 34 / 48)
        self.assertAlmostEqual(closed.metrics.macro_precision, (9 / 16 + 16 / 23 + 1) / 3)
        self.assertAlmostEqual(closed.metrics.macro_recall, 34 / 48)
        self.assertAlmostEqual(closed.metrics.macro_f1, (9 / 16 + 32 / 39 + 18 / 25) / 3)

    def test_persisted_references_and_transforms_are_used_without_self_exclusion(self):
        runner = SklearnFinalOpenSetRunner(self.model)
        output = runner.infer(self.data.known_X, self.data.known_y, self.data.unknown_X)
        Z = self.model.parts.scaler.transform(self.model.parts.selector.transform(self.data.known_X))
        expected = self.model.parts.reference.score(Z).confidence
        self.assertTrue(np.array_equal(output.known.confidence[0][1], expected[0][1]))
        self.assertTrue(np.all(output.known.confidence[0][1] <= 0))
        self.assertEqual(sum(row.support for row in output.closed_set.per_class), 48)
        for index, row in enumerate(output.closed_set.per_class):
            matrix = np.array(output.closed_set.confusion_matrix)
            tp, support, predicted = matrix[index, index], matrix[index].sum(), matrix[:, index].sum()
            self.assertEqual(row.support, support)
            self.assertAlmostEqual(row.precision, tp / predicted if predicted else 0.)
            self.assertAlmostEqual(row.recall, tp / support if support else 0.)

    def test_condition_aggregation_comparison_and_vote_counts(self):
        runner, _ = self.fake_runner()
        report = FinalOpenSetEvaluator(runner).evaluate(self.data, self.context)
        for condition in report.conditions:
            self.assertEqual((condition.known_source_samples, condition.unknown_source_samples), (24, 24))
            self.assertEqual((condition.known_windows, condition.unknown_windows), (3, 3))
            self.assertAlmostEqual(condition.known_frame_accuracy, 17 / 24)
            self.assertEqual(condition.known_acceptance_rate, 2 / 3)
            self.assertEqual(condition.known_rejection_rate, 1 / 3)
            self.assertEqual(condition.known_correct_identification_rate, 1 / 3)
        self.assertEqual(report.known_vote_stability.unanimous_windows, 6)
        self.assertEqual(report.known_vote_stability.rates.unanimous_rate, 1.)
        self.assertAlmostEqual(report.delta_final_minus_development.roc_auc, report.final_holdout.roc_auc - report.development_candidate.roc_auc)
        self.assertIn("not final generalization", report.calibration_warning)
        self.assertEqual(model_fingerprints(self.bundle[2]), self.context.model_fingerprints)

    def test_window_seven_median_majority_and_summed_score_tie_diagnostics(self):
        runner, output = self.fake_runner()
        plan = build_windows(self.data.known_samples, 7)
        scores = output.known.confidence[0][1].copy()
        # Median accepts this window while its skewed mean would reject it.
        scores[list(plan.windows[0].indices)] = self.context.threshold + np.array([-100., -90., -80., 1., 2., 3., 4.])
        predicted, decisions = output.known.predicted.copy(), output.known.decisions.copy()
        majority = list(plan.windows[1].indices)
        predicted[majority] = np.array([1, 1, 1, 1, 0, 0, 0])
        tie = list(plan.windows[2].indices)
        predicted[tie] = np.array([0, 0, 0, 1, 1, 1, 2])
        decisions[tie] = np.eye(3)[predicted[tie]]
        decisions[tie, 1] += .1
        output.known = FrameInference(predicted, decisions, (("mean_5nn", scores),))
        metrics, matrix = classification_measurements(self.data.known_y, predicted, 3)
        output.closed_set = replace(output.closed_set, metrics=metrics, confusion_matrix=matrix)
        report = FinalOpenSetEvaluator(runner).evaluate(self.data, self.context)
        self.assertEqual(report.operational_metrics.known_acceptance_rate, 4 / 6)
        self.assertEqual((report.known_vote_stability.unanimous_windows, report.known_vote_stability.majority_not_unanimous_windows,
                          report.known_vote_stability.tied_windows), (4, 1, 1))
        self.assertEqual(tuple(vars(report.known_vote_stability.rates).values()), (4 / 6, 1 / 6, 1 / 6))
        windows = aggregate_windows(plan, predicted, decisions, (("mean_5nn", scores),))
        self.assertEqual(windows.predicted[2], 1)
        decisions[tie] = np.zeros((7, 3))
        tied_again = aggregate_windows(plan, predicted, decisions, (("mean_5nn", scores),))
        self.assertEqual(tied_again.predicted[2], 0)
