from collections import Counter
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from face_recognition.application import evaluate_v3_mahalanobis_calibration as study
from face_recognition.application.rejection_metrics import calibrate_threshold
from face_recognition.domain.models.rejection import RejectionError
from face_recognition.infrastructure.classifiers.sklearn_v3_same_model import SklearnV3SameModelRunner
from face_recognition.infrastructure.features.mahalanobis import PredictedClassMahalanobisScorer
from face_recognition.infrastructure.persistence.v3_mahalanobis_calibration_report import (
    mahalanobis_calibration_report_to_json, read_mahalanobis_selection_report,
)
from tests.v3_mahalanobis_fakes import RecordingSameModelRunner, write_selection_report
from tests.v3_novelty_fakes import novelty_data


class V3MahalanobisCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory(prefix="synthetic_v3_same_model_")
        cls.folder = Path(cls.temporary.name)
        cls.data, _, _ = novelty_data(cls.folder)
        cls.source_path = write_selection_report(cls.data, cls.folder / "selection.json")
        cls.selection = read_mahalanobis_selection_report(cls.source_path, cls.data.protocol)
        cls.runner = RecordingSameModelRunner()
        cls.thresholds = []
        def calibration(*args, **kwargs):
            result = calibrate_threshold(*args, **kwargs)
            cls.runner.events.append(("calibrate", cls.runner.fit_count, len(args[0]), len(args[2]), result.threshold))
            cls.thresholds.append(result.threshold)
            return result
        with patch.object(study, "calibrate_threshold", calibration):
            cls.report = study.V3MahalanobisCalibrationEvaluator(cls.runner).evaluate(cls.data, "a" * 64, cls.selection)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_cyclic_assignment_exact_and_each_session_test_calibration_once_fit_twice(self):
        roles = study.cyclic_session_roles(self.data.protocol)
        self.assertEqual(roles, study.cyclic_session_roles(self.data.protocol))
        a, b, c, d = self.data.protocol.specification.known_sessions
        self.assertEqual([(r.known_fit_sessions, r.known_calibration_session, r.known_test_session) for r in roles],
                         [((c, d), b, a), ((a, d), c, b), ((a, b), d, c), ((b, c), a, d)])
        self.assertEqual(Counter(r.known_test_session for r in roles), Counter({s: 1 for s in (a, b, c, d)}))
        self.assertEqual(Counter(r.known_calibration_session for r in roles), Counter({s: 1 for s in (a, b, c, d)}))
        self.assertEqual(Counter(s for r in roles for s in r.known_fit_sessions), Counter({s: 2 for s in (a, b, c, d)}))

    def test_one_fit_same_model_scores_calibration_and_test_and_threshold_is_unchanged(self):
        self.assertEqual(self.runner.fit_count, 4)
        self.assertEqual(len(self.runner.model_calls), 16)
        self.assertEqual(self.thresholds, [r.threshold for r in self.report.outer_results])
        for i, row in enumerate(self.report.outer_results):
            calls = self.runner.model_calls[i * 4:i * 4 + 4]
            self.assertEqual(len({call[1] for call in calls}), 1)
            self.assertEqual({call[0] for call in calls}, {i + 1})
            self.assertEqual(calls[0][2:], (self.data.protocol.specification.known_ids, (row.roles.known_calibration_session,)))
            self.assertEqual(calls[1][2], row.roles.unknown_calibration_ids)
            self.assertEqual(calls[2][2:], (self.data.protocol.specification.known_ids, (row.roles.known_test_session,)))
            self.assertEqual(calls[3][2], row.roles.unknown_test_ids)
            self.assertEqual(row.model_sha256_before_calibration, row.model_sha256_after_test)
            self.assertEqual((row.fit_calls, row.same_model_identity_verified, row.no_refit_after_calibration), (1, True, True))
            self.assertEqual(row.threshold_transport, "none_same_model_calibration")
            fits = [e for e in self.runner.events if e[0] == "fit"]
            self.assertEqual(fits[i][1], row.roles.known_fit_sessions)
            self.assertFalse(set(fits[i][1]) & {row.roles.known_calibration_session, row.roles.known_test_session})
            self.assertEqual(fits[i][2], self.data.protocol.specification.known_ids)
            calibration_index = next(j for j, e in enumerate(self.runner.events) if e[:2] == ("calibrate", i + 1))
            self.assertEqual([e[0] for e in self.runner.events[calibration_index + 1:calibration_index + 3]], ["infer", "infer"])

    def test_all_six_unknown_development_ids_calibrate_using_windows_not_frames(self):
        for i, row in enumerate(self.report.outer_results):
            outer = self.data.protocol.outer_folds[i]
            self.assertEqual(row.roles.unknown_calibration_ids, outer.unknown_development_ids)
            self.assertEqual(len(row.roles.unknown_calibration_ids), 6)
            self.assertFalse(set(row.roles.unknown_calibration_ids) & set(outer.unknown_test_ids))
            self.assertEqual((row.known_calibration_counts.windows, row.unknown_calibration_counts.windows), (26, 156))
            cal = next(e for e in self.runner.events if e[:2] == ("calibrate", i + 1))
            self.assertEqual(cal[2:4], (26, 156))
            self.assertNotEqual(cal[2], row.known_calibration_counts.source_samples)
            self.assertEqual(row.known_test_counts.discarded_samples, 26)
            self.assertEqual(row.unknown_test_counts.discarded_samples, 52)

    def test_outer_uses_calibrated_threshold_not_diagnostic_eer(self):
        original_pair = study.evaluate_pair
        seen = []
        def calibration(*args, **kwargs):
            result = calibrate_threshold(*args, **kwargs)
            # Deliberately different finite diagnostic test threshold, to prove
            # outer evaluation never replaces it with its own EER cutpoint.
            return replace(result, threshold=100.)
        def evaluate_pair(known, unknown, threshold):
            seen.append(threshold)
            return original_pair(known, unknown, threshold)
        with patch.object(study, "calibrate_threshold", calibration), patch.object(study, "evaluate_pair", evaluate_pair):
            report = study.V3MahalanobisCalibrationEvaluator(RecordingSameModelRunner()).evaluate(self.data, "a" * 64, self.selection)
        self.assertEqual(seen, [100.] * 4)
        self.assertTrue(all(r.operational_metrics.known_rejection_rate == 1. for r in report.outer_results))
        self.assertTrue(all(r.diagnostics.equal_error_point.threshold != 100. for r in report.outer_results))
        tied = calibrate_threshold(np.array([-1., -2.]), np.ones(2, dtype=bool), np.array([-5., -4.]), allow_negative=True)
        self.assertEqual(tied.threshold, -2.)

    def test_refit_identical_state_mutation_or_component_replacement_is_rejected(self):
        fit_population = self.data.known(study.cyclic_session_roles(self.data.protocol)[0].known_fit_sessions)
        for defect in ("refit", "mutate", "replace_identity"):
            class BadRunner(RecordingSameModelRunner):
                def __init__(self):
                    super().__init__()
                    self.calls, self.identity_generation = 0, 0
                def infer(self, model, population):
                    self.calls += 1
                    if self.calls == 3:
                        if defect == "refit": self.fit(fit_population, 2)
                        elif defect == "mutate": model.offset += 1.
                        else: self.identity_generation += 1
                    return super().infer(model, population)
                def model_identity(self, model):
                    return (id(model), self.identity_generation)
            with self.subTest(defect=defect), self.assertRaisesRegex(RejectionError, "refitted, replaced or mutated"):
                study.V3MahalanobisCalibrationEvaluator(BadRunner()).evaluate(self.data, "a" * 64, self.selection)

    def test_phase_authorization_rejects_outer_test_as_calibration_and_duplicate_scoring(self):
        roles = study.cyclic_session_roles(self.data.protocol)[0]
        fold = study._FitOnceFold(RecordingSameModelRunner(), self.data, roles)
        for population in (self.data.known((roles.known_test_session,)), self.data.unknown(roles.unknown_test_ids)):
            with self.assertRaises(RejectionError): fold.infer(population, calibration=True)
        known, unknown = self.data.known((roles.known_calibration_session,)), self.data.unknown(roles.unknown_calibration_ids)
        fold.infer(known, calibration=True)
        with self.assertRaises(RejectionError): fold.infer(known, calibration=True)
        with self.assertRaises(RejectionError): fold.freeze_threshold(0.)
        fold.infer(unknown, calibration=True)
        with self.assertRaises(RejectionError): fold.freeze_threshold(float("nan"))
        fold.freeze_threshold(-2.)
        with self.assertRaises(RejectionError): fold.freeze_threshold(-3.)
        with self.assertRaises(RejectionError): fold.infer(known, calibration=False)

    def test_real_components_fit_only_two_known_sessions_and_never_refit_during_scoring(self):
        runner = SklearnV3SameModelRunner()
        roles = study.cyclic_session_roles(self.data.protocol)
        originals = {cls: cls.fit for cls in (SelectKBest, StandardScaler, SVC, PredictedClassMahalanobisScorer)}
        counts = Counter()
        active = {"population": None}
        original_runner_fit = runner.fit
        def fit(population, identities, methods):
            expected = self.data.known(roles[runner.fit_count].known_fit_sessions)
            np.testing.assert_array_equal(population.X, expected.X)
            active["population"] = population
            try: return original_runner_fit(population, identities, methods)
            finally: active["population"] = None
        def spy(cls):
            def method(self, X, y=None, *args, **kwargs):
                population = active["population"]
                if population is None: raise AssertionError("Component fit after freeze")
                np.testing.assert_array_equal(y, population.y)
                self_expected = 1434 if cls is SelectKBest else 448
                if X.shape != (len(population.y), self_expected): raise AssertionError("Wrong fitting population")
                counts[cls] += 1
                return originals[cls](self, X, y, *args, **kwargs)
            return method
        from contextlib import ExitStack
        with ExitStack() as stack, warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning); warnings.simplefilter("ignore", RuntimeWarning)
            stack.enter_context(patch.object(runner, "fit", fit))
            for cls in originals: stack.enter_context(patch.object(cls, "fit", spy(cls)))
            first = study.V3MahalanobisCalibrationEvaluator(runner).evaluate(self.data, "a" * 64, self.selection)
        self.assertEqual(runner.fit_count, 4)
        self.assertTrue(all(counts[cls] == 4 for cls in originals))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning); warnings.simplefilter("ignore", RuntimeWarning)
            second = study.V3MahalanobisCalibrationEvaluator(SklearnV3SameModelRunner()).evaluate(self.data, "a" * 64, self.selection)
        self.assertEqual(mahalanobis_calibration_report_to_json(first), mahalanobis_calibration_report_to_json(second))

    def test_aggregate_ranges_and_deterministic_synthetic_report(self):
        second = study.V3MahalanobisCalibrationEvaluator(RecordingSameModelRunner()).evaluate(self.data, "a" * 64, self.selection)
        self.assertEqual(mahalanobis_calibration_report_to_json(self.report), mahalanobis_calibration_report_to_json(second))
        self.assertEqual(self.report.mean_closed_set.accuracy, 7 / 8)
        self.assertEqual(self.report.std_closed_set.accuracy, 0.)
        self.assertEqual(self.report.far_range, (0., 0.))
        self.assertEqual(self.report.frr_range, (0., 0.))
        self.assertIn("refinement", self.report.methodology)
        self.assertIn("no additional untouched holdout", self.report.limitation)

    def test_prior_outer_metrics_and_thresholds_are_not_calibration_inputs(self):
        source = json.loads(self.source_path.read_text(encoding="utf-8"))
        for row in source["outer_results"]:
            row["threshold"] = -1e20
            row["operational_metrics"] = {"unknown_false_acceptance_rate": 0.}
        path = self.folder / "alternative_synthetic_selection.json"
        path.write_text(json.dumps(source), encoding="utf-8")
        selection = read_mahalanobis_selection_report(path, self.data.protocol)
        second = study.V3MahalanobisCalibrationEvaluator(RecordingSameModelRunner()).evaluate(self.data, "a" * 64, selection)
        self.assertEqual(self.report.outer_results, second.outer_results)
