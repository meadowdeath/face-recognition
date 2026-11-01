from dataclasses import replace
from pathlib import Path
from statistics import pstdev
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from face_recognition.application.refit_v2_open_set_model import V2OpenSetRefitter, threshold_stability, validate_direct_transfer
from face_recognition.application.rejection_score_analysis import analyze_scores
from face_recognition.application.temporal_windows import aggregate_windows, build_windows
from face_recognition.domain.models.open_set_model import OpenSetModelError
from face_recognition.domain.models.open_set_config import DevelopmentReportProvenance, FrozenDevelopmentEvidence, FrozenOpenSetConfiguration
from face_recognition.infrastructure.classifiers.sklearn_v2_open_set_refit import SklearnV2OpenSetRefitRunner
from face_recognition.infrastructure.features.rejection_scores import decision_top1
from tests.v2_open_set_refit_fakes import v2_refit_fixture, load_v2_refit_fixture


class V2TransferTests(unittest.TestCase):
    def test_fold_threshold_statistics(self):
        result = threshold_stability((1., 2., 3.))
        self.assertEqual((result.minimum, result.maximum, result.range, result.mean, result.median), (1., 3., 2., 2., 2.))
        self.assertEqual(result.population_std, pstdev((1., 2., 3.)))
        for values in ((1., 2.), (1., 2., np.nan)):
            with self.assertRaises(OpenSetModelError):
                threshold_stability(values)

    def test_direct_transfer_deterministic_warning_and_failure(self):
        known, unknown, stable = np.ones(4), np.zeros(4), threshold_stability((.4, .5, .6))
        result = validate_direct_transfer(known, unknown, known + .1, unknown + .1, stable)
        self.assertEqual(result, validate_direct_transfer(known, unknown, known + .1, unknown + .1, stable))
        self.assertEqual(result.status, "passed")
        self.assertAlmostEqual(result.known_mean_shift, .1)
        self.assertFalse(result.warning)
        warning = validate_direct_transfer(known, unknown, known + 2, unknown, stable)
        self.assertTrue(warning.warning)
        self.assertEqual(validate_direct_transfer(known, unknown, known + 3, unknown, stable).status, "passed")
        with self.assertRaisesRegex(OpenSetModelError, "new DEVELOPMENT protocol"):
            validate_direct_transfer(known, unknown, known + 4, unknown, stable)
        with self.assertRaises(OpenSetModelError):
            validate_direct_transfer(known, unknown, known, unknown, threshold_stability((.5, .5, 20.)))

    def test_nonfinite_and_forged_stability_rejected(self):
        stable = threshold_stability((1., 1., 1.))
        for bad in (np.nan, np.inf, -np.inf):
            with self.assertRaises(ValueError):
                validate_direct_transfer(np.array([bad]), np.zeros(1), np.ones(1), np.zeros(1), stable)
        with self.assertRaises(OpenSetModelError):
            validate_direct_transfer(np.ones(1), np.zeros(1), np.ones(1), np.zeros(1), replace(stable, range=np.nan))

    def test_top1_maximum_orientation(self):
        values = decision_top1(np.array([[1., 3., 2.], [-3., -1., -2.]]))
        self.assertTrue(bool(np.array_equal(values, [3., -1.])))
        with self.assertRaises(ValueError):
            decision_top1(np.array([[np.nan, 2.]]))


class V2OpenSetRefitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory(prefix="synthetic_v2_refit_")
        cls.fixture = v2_refit_fixture(Path(cls.temporary.name))
        cls.data = load_v2_refit_fixture(cls.fixture)
        cls.config, cls.digest = cls.fixture[-2:]
        cls.runner = SklearnV2OpenSetRefitRunner()
        cls.result = V2OpenSetRefitter(cls.runner).run(cls.config, cls.digest, cls.data)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_final_structure_classes_and_no_reference(self):
        s, p = self.result.summary, self.result.parts
        self.assertEqual((s.known_training_samples, s.unknown_unique_source_samples, s.identity_count), (144, 80, 3))
        self.assertEqual((s.original_feature_count, s.selected_feature_count), (1434, 448))
        self.assertTrue(bool(np.array_equal(p.svm.classes_, np.arange(3))))
        self.assertEqual(p.class_ids, self.data.source.specification.known_ids)
        self.assertFalse(p.svm.probability)
        self.assertFalse(hasattr(p, "reference"))
        self.assertEqual(p.svm.shape_fit_, (144, 448))
        self.assertNotIn("array(", repr(self.result))

    def test_four_known_only_fits_and_held_sessions_excluded(self):
        inputs, step_fits = [], []
        original = self.runner.fit_model

        def fit(config, X, y, ids):
            i = len(inputs)
            if i < 3:
                session = config.known_cross_session_folds[i].calibration_session
                held = np.array([s.session_id == session for s in self.data.known_samples])
                self.assertTrue(bool(np.array_equal(X, self.data.known_X[~held])))
                self.assertTrue(bool(np.array_equal(y, self.data.known_y[~held])))
            else:
                self.assertTrue(bool(np.array_equal(X, self.data.known_X)))
            inputs.append(len(X))
            return original(config, X, y, ids)

        def spy(kind):
            old = kind.fit
            def fit_step(instance, X, y=None, **kwargs):
                step_fits.append((kind.__name__, len(X)))
                return old(instance, X, y, **kwargs)
            return fit_step

        with patch.object(self.runner, "fit_model", fit), patch.object(SelectKBest, "fit", spy(SelectKBest)), \
                patch.object(StandardScaler, "fit", spy(StandardScaler)), patch.object(SVC, "fit", spy(SVC)):
            V2OpenSetRefitter(self.runner).run(self.config, self.digest, self.data)
        self.assertEqual(inputs, [96, 96, 96, 144])
        for name in ("SelectKBest", "StandardScaler", "SVC"):
            self.assertEqual([n for k, n in step_fits if k == name], inputs)

    def test_oof_temporal_populations_threshold_not_resubstitution(self):
        calibration_inputs = []
        inferred = []
        original_infer = self.runner.infer

        def infer(parts, X):
            frames = original_infer(parts, X)
            inferred.append(frames)
            return frames

        def analyze(k, c, u):
            calibration_inputs.append((k.copy(), c.copy(), u.copy()))
            return analyze_scores(k, c, u)

        with patch.object(self.runner, "infer", infer), patch("face_recognition.application.refit_v2_open_set_model.analyze_scores", analyze):
            result = V2OpenSetRefitter(self.runner).run(self.config, self.digest, self.data)
        self.assertEqual(len(calibration_inputs), 4)  # Three diagnostics, one pooled operational choice; none after final fit.
        for i in range(3):
            held = np.array([s.session_id == self.config.known_cross_session_folds[i].calibration_session for s in self.data.known_samples])
            samples = tuple(s for s, yes in zip(self.data.known_samples, held) if yes)
            plan = build_windows(samples, 7)
            frames = inferred[2 * i]
            observations = aggregate_windows(plan, frames.predicted, frames.decisions, frames.confidence)
            self.assertTrue(bool(np.array_equal(calibration_inputs[i][0], observations.confidence[0][1])))
            self.assertEqual(len(calibration_inputs[i][0]), 6)
            self.assertEqual(len(calibration_inputs[i][2]), 10)
            for w in plan.windows:
                self.assertEqual(len({(samples[j].person_id, samples[j].session_id, samples[j].condition) for j in w.indices}), 1)
                self.assertEqual(len(w.indices), 7)
        K, C, U = calibration_inputs[-1]
        self.assertTrue(bool(np.array_equal(K, np.concatenate([v[0] for v in calibration_inputs[:3]]))))
        self.assertTrue(bool(np.array_equal(U, np.concatenate([v[2] for v in calibration_inputs[:3]]))))
        self.assertEqual(result.summary.analysis, analyze_scores(K, C, U))
        self.assertEqual((len(K), len(U)), (18, 30))

    def test_counts_distributions_and_threshold_stability(self):
        s = self.result.summary
        self.assertEqual((s.known_oof_counts.source_samples, s.known_oof_counts.used_samples, s.known_oof_counts.discarded_samples), (144, 126, 18))
        self.assertEqual((s.unknown_unique_counts.source_samples, s.unknown_unique_counts.windows), (80, 10))
        self.assertEqual((s.unknown_fold_observation_counts.source_samples, s.unknown_fold_observation_counts.windows), (240, 30))
        self.assertEqual(s.threshold_stability, threshold_stability(f.analysis.calibration.threshold for f in s.folds))
        for f in s.folds:
            self.assertEqual(f.analysis.known_distribution.count, 6)
            self.assertEqual(f.analysis.unknown_distribution.count, 10)
        self.assertEqual(s.transfer_validation.status, "passed")
        self.assertEqual(s.v2_final_evaluation, "not_performed")
        self.assertFalse(s.future_known_session_payloads_accessed or s.future_unknown_payloads_accessed)

    def test_unsafe_transfer_stops_without_aligned_threshold(self):
        with patch("face_recognition.application.refit_v2_open_set_model.validate_direct_transfer", side_effect=OpenSetModelError("Frozen V2 direct-transfer assumption failed")):
            with self.assertRaisesRegex(OpenSetModelError, "assumption failed"):
                V2OpenSetRefitter(self.runner).run(self.config, self.digest, self.data)

    def test_wrong_config_or_provenance_fails_before_fit(self):
        v1 = FrozenOpenSetConfiguration(DevelopmentReportProvenance("a" * 64),
                                         FrozenDevelopmentEvidence(.8, .2, self.config.development_evidence.metrics, 18, 10))
        with patch.object(self.runner, "fit_model") as fit:
            for config, digest in ((None, self.digest), (v1, self.digest), (self.config, "bad"),
                                   (replace(self.config, provenance=replace(self.config.provenance, original_identity_split_sha256="c" * 64)), self.digest)):
                with self.assertRaises(OpenSetModelError):
                    V2OpenSetRefitter(self.runner).run(config, digest, self.data)
            fit.assert_not_called()

    def test_unsafe_fold_scales_stop_before_final_refit(self):
        original = self.runner.fit_model
        with patch.object(self.runner, "fit_model", wraps=original) as fits, \
                patch("face_recognition.application.refit_v2_open_set_model.calibration_scale",
                      side_effect=OpenSetModelError("Frozen V2 transfer assumption failed")):
            with self.assertRaises(OpenSetModelError):
                V2OpenSetRefitter(self.runner).run(self.config, self.digest, self.data)
            self.assertEqual(fits.call_count, 3)


if __name__ == "__main__":
    unittest.main()
