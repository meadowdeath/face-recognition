from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from face_recognition.application import evaluate_v3_pose_representation as experiment
from face_recognition.application.rejection_metrics import calibrate_threshold
from face_recognition.domain.models.dataset import CONDITIONS
from face_recognition.domain.models.processed_sample import BASELINE_REPRESENTATION, POSE_REPRESENTATION
from face_recognition.domain.models.rejection import RejectionError
from face_recognition.domain.models.v3_pose_representation import POSE_DELTA_CONDITIONS
from face_recognition.infrastructure.features.pose_landmark_normalizer import POSE_NORMALIZATION
from face_recognition.infrastructure.persistence.v3_pose_representation import pose_representation_report_to_json
from tests.pose_representation_fakes import paired_pose_data
from tests.v3_mahalanobis_fakes import RecordingSameModelRunner


class V3PoseRepresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory(prefix="synthetic_v3_pose_comparison_")
        cls.baseline, cls.candidate, cls.catalog, _, _ = paired_pose_data(Path(cls.temporary.name))
        cls.runners, cls.threshold_events = [], []
        def factory():
            runner = RecordingSameModelRunner()
            cls.runners.append(runner)
            return runner
        def calibration(*args, **kwargs):
            value = calibrate_threshold(*args, **kwargs)
            runner = cls.runners[-1]
            cls.threshold_events.append((len(cls.runners) - 1, runner.fit_count, len(args[0]), len(args[2]), value.threshold))
            runner.events.append(("calibrated", runner.fit_count, value.threshold))
            return value
        with patch.object(experiment, "calibrate_threshold", calibration):
            cls.report = experiment.V3PoseRepresentationEvaluator(factory).evaluate(
                cls.baseline, cls.candidate, "a" * 64, cls.catalog, POSE_NORMALIZATION)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_representations_share_exact_fold_roles_and_only_two_known_sessions_fit(self):
        baseline, candidate = self.report.results
        self.assertEqual([f.evaluation.roles for f in baseline.folds], [f.evaluation.roles for f in candidate.folds])
        self.assertEqual(len(self.runners), 2)
        for runner, result in zip(self.runners, self.report.results):
            fits = [e for e in runner.events if e[0] == "fit"]
            self.assertEqual(runner.fit_count, 4)
            for fit, fold in zip(fits, result.folds):
                roles = fold.evaluation.roles
                self.assertEqual(fit[1], roles.known_fit_sessions)
                self.assertEqual(len(fit[1]), 2)
                self.assertEqual(fit[2], self.baseline.protocol.specification.known_ids)
                self.assertEqual(fit[4], ("mahalanobis_distance",))
                self.assertFalse(set(fit[1]) & {roles.known_calibration_session, roles.known_test_session})

    def test_classifier_mahalanobis_window_and_threshold_policy_are_identical(self):
        first, second = (r.configuration for r in self.report.results)
        self.assertEqual(first, replace(second, representation=BASELINE_REPRESENTATION))
        self.assertEqual(dict(first.svm_parameters)["C"], 4)
        self.assertEqual(dict(first.svm_parameters)["gamma"], .001)
        self.assertIs(dict(first.svm_parameters)["probability"], False)
        self.assertEqual(first.selected_features, 448)
        self.assertEqual(first.novelty_method, "mahalanobis_distance")
        self.assertIn("LedoitWolf", first.covariance)
        self.assertEqual(first.window_size, 7)
        self.assertEqual(first.confidence_aggregation, "median")
        self.assertFalse(first.consistency_gate)
        for result in self.report.results:
            for fold in result.folds:
                row = fold.evaluation
                self.assertEqual(row.known_test_counts.used_samples, row.known_test_counts.windows * 7)
                self.assertEqual(row.known_test_counts.discarded_samples, 26)
                self.assertEqual(row.threshold_transport, "none_same_model_calibration")

    def test_calibration_uses_window_scores_and_threshold_remains_frozen_without_refit(self):
        self.assertEqual(len(self.threshold_events), 8)
        for index, result in enumerate(self.report.results):
            runner = self.runners[index]
            for fold_index, fold in enumerate(result.folds):
                row = fold.evaluation
                event = self.threshold_events[index * 4 + fold_index]
                self.assertEqual(event, (index, fold_index + 1, row.known_calibration_counts.windows, row.unknown_calibration_counts.windows, row.threshold))
                self.assertNotEqual(event[2], row.known_calibration_counts.source_samples)
                calls = runner.model_calls[fold_index * 4:fold_index * 4 + 4]
                self.assertEqual(len({call[1] for call in calls}), 1)
                self.assertEqual({call[0] for call in calls}, {fold_index + 1})
                self.assertEqual(calls[0][3], (row.roles.known_calibration_session,))
                self.assertEqual(calls[1][2], row.roles.unknown_calibration_ids)
                self.assertEqual(calls[2][3], (row.roles.known_test_session,))
                self.assertEqual(calls[3][2], row.roles.unknown_test_ids)
                frozen = next(i for i, e in enumerate(runner.events) if e[:2] == ("calibrated", fold_index + 1))
                self.assertEqual([e[0] for e in runner.events[frozen + 1:frozen + 3]], ["infer", "infer"])
                self.assertEqual(row.model_sha256_before_calibration, row.model_sha256_after_test)
                self.assertTrue(row.no_refit_after_calibration and row.same_model_identity_verified)

    def test_operational_threshold_is_not_replaced_with_outer_eer(self):
        def calibration(*args, **kwargs):
            return replace(calibrate_threshold(*args, **kwargs), threshold=100.)
        with patch.object(experiment, "calibrate_threshold", calibration):
            report = experiment.V3PoseRepresentationEvaluator(RecordingSameModelRunner).evaluate(
                self.baseline, self.candidate, "a" * 64, self.catalog, POSE_NORMALIZATION)
        for result in report.results:
            for fold in result.folds:
                self.assertEqual(fold.evaluation.threshold, 100.)
                self.assertEqual(fold.evaluation.operational_metrics.known_rejection_rate, 1.)
                self.assertNotEqual(fold.evaluation.diagnostics.equal_error_point.threshold, 100.)

    def test_all_13_conditions_explicitly_include_pitch_down_and_required_deltas(self):
        self.assertEqual(tuple(r.condition for r in self.report.condition_comparison), CONDITIONS)
        self.assertIn("pitch_down", POSE_DELTA_CONDITIONS)
        deltas = dict(self.report.pose_distance_frr_deltas)
        self.assertEqual(set(deltas), {"yaw_left", "yaw_right", "pitch_up", "pitch_down", "roll_left", "roll_right", "near", "far"})
        for condition in self.report.condition_comparison:
            if condition.condition in deltas:
                self.assertEqual(deltas[condition.condition], condition.candidate_frr - condition.baseline_frr)
        for result in self.report.results:
            for fold in result.folds:
                self.assertEqual(tuple(r.label for r in fold.known_per_condition), CONDITIONS)
                self.assertEqual(tuple(r.label for r in fold.unknown_per_condition), CONDITIONS)
                pitch = next(r for r in fold.known_per_condition if r.label == "pitch_down")
                self.assertEqual(pitch.counts.total_windows, 2)
                self.assertIsNotNone(pitch.scores.mean)
                self.assertIsNotNone(pitch.scores.median)
            self.assertEqual(len(result.diagnostics.known_session_severity), 4)
            self.assertEqual(len(result.diagnostics.unknown_identity_severity), 8)

    def test_ranking_priority_and_baseline_wins_exact_tie_deterministically(self):
        base, pose = self.report.results
        self.assertEqual(experiment.rank_representations((pose, base))[0], BASELINE_REPRESENTATION)
        self.assertEqual(self.report.selected_development_representation, BASELINE_REPRESENTATION)
        # Make each earlier criterion tie before checking the next one.
        base = replace(base, mean_metrics=replace(base.mean_metrics, binary_balanced_accuracy=.8,
            known_rejection_rate=.2, unknown_false_acceptance_rate=.2), mean_closed_set=replace(base.mean_closed_set, macro_f1=.8))
        pose = replace(pose, mean_metrics=base.mean_metrics, mean_closed_set=base.mean_closed_set)
        changes = (
            replace(pose, mean_metrics=replace(pose.mean_metrics, binary_balanced_accuracy=.81, known_rejection_rate=.9)),
            replace(pose, mean_metrics=replace(pose.mean_metrics, known_rejection_rate=.19, unknown_false_acceptance_rate=.9)),
            replace(pose, mean_metrics=replace(pose.mean_metrics, unknown_false_acceptance_rate=.19)),
            replace(pose, mean_closed_set=replace(pose.mean_closed_set, macro_f1=.81)),
        )
        for changed in changes:
            self.assertEqual(experiment.rank_representations((base, changed))[0], POSE_REPRESENTATION)
            self.assertEqual(experiment.rank_representations((changed, base))[0], POSE_REPRESENTATION)

    def test_population_metadata_representation_or_protocol_mismatch_rejected(self):
        for candidate in (replace(self.candidate, representation=BASELINE_REPRESENTATION),
                          replace(self.candidate, samples=tuple(replace(s, detector_timestamp_ms=s.detector_timestamp_ms + 1) for s in self.candidate.samples))):
            with self.assertRaises(RejectionError): experiment.validate_representation_pair(self.baseline, candidate)
        bad_catalog = replace(self.catalog, condition_files=self.catalog.condition_files[:-1])
        with self.assertRaises(RejectionError):
            experiment.V3PoseRepresentationEvaluator(RecordingSameModelRunner).evaluate(
                self.baseline, self.candidate, "a" * 64, bad_catalog, POSE_NORMALIZATION)

    def test_identical_inputs_produce_byte_identical_aggregate_report(self):
        second = experiment.V3PoseRepresentationEvaluator(RecordingSameModelRunner).evaluate(
            self.baseline, self.candidate, "a" * 64, self.catalog, POSE_NORMALIZATION)
        self.assertEqual(pose_representation_report_to_json(self.report), pose_representation_report_to_json(second))

    def test_real_component_spies_use_only_authorized_known_fit_populations(self):
        from contextlib import ExitStack
        import warnings
        from sklearn.feature_selection import SelectKBest
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        from face_recognition.infrastructure.features.mahalanobis import PredictedClassMahalanobisScorer
        from face_recognition.infrastructure.classifiers.sklearn_v3_same_model import SklearnV3SameModelRunner
        active, calls = {}, []
        class SpyRunner(SklearnV3SameModelRunner):
            def fit(self, population, identities, methods):
                active["population"] = population
                try: return super().fit(population, identities, methods)
                finally: active.clear()
        originals = {cls: cls.fit for cls in (SelectKBest, StandardScaler, SVC, PredictedClassMahalanobisScorer)}
        def spy(cls):
            def fit(component, X, y=None, *args, **kwargs):
                population = active["population"]  # Failure if fitting after calibration.
                self.assertTrue(np.all(population.y >= 0))
                self.assertEqual(len({s.session_id for s in population.samples}), 2)
                np.testing.assert_array_equal(y, population.y)
                self.assertEqual(X.shape, (len(population.y), 1434 if cls is SelectKBest else 448))
                if cls is SVC:
                    self.assertEqual((component.kernel, component.C, component.gamma, component.probability), ("rbf", 4, .001, False))
                calls.append(cls)
                return originals[cls](component, X, y, *args, **kwargs)
            return fit
        with ExitStack() as stack, warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning); warnings.simplefilter("ignore", RuntimeWarning)
            for cls in originals: stack.enter_context(patch.object(cls, "fit", spy(cls)))
            report = experiment.V3PoseRepresentationEvaluator(SpyRunner).evaluate(
                self.baseline, self.candidate, "a" * 64, self.catalog, POSE_NORMALIZATION)
        self.assertEqual(len(report.results), 2)
        for cls in originals: self.assertEqual(calls.count(cls), 8)
