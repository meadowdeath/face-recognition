from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from face_recognition.application import train_v3_deployable_model as training
from face_recognition.application.rejection_metrics import calibrate_threshold
from face_recognition.domain.models.rejection import RejectionError
from face_recognition.infrastructure.classifiers.v3_fitted_integrity import fitted_value_sha256
from tests.v3_deployable_fakes import RecordingDeployableRunner, deployable_fixture


class TrainV3DeployableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory(prefix="synthetic_v3_train_")
        cls.data, cls.config, cls.provenance, *_ = deployable_fixture(Path(cls.temporary.name))
        cls.runner = RecordingDeployableRunner()
        cls.calibration_inputs = []
        def calibration(known, correct, unknown, **kwargs):
            cls.calibration_inputs.append((known.copy(), correct.copy(), unknown.copy()))
            result = calibrate_threshold(known, correct, unknown, **kwargs)
            cls.runner.events.append(("calibrate", result.threshold))
            return result
        with patch.object(training, "calibrate_threshold", calibration):
            cls.result = training.TrainV3DeployableModel(cls.runner).train(cls.data, cls.config, cls.provenance)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_only_first_three_known_sessions_fit_latest_known_only_calibrates(self):
        spec = self.data.protocol.specification
        fit = [e for e in self.runner.events if e[0] == "fit"]
        self.assertEqual(len(fit), 1)
        self.assertEqual(fit[0][1:3], (spec.known_sessions[:3], spec.known_ids))
        self.assertEqual(fit[0][4], ("mahalanobis_distance",))
        self.assertEqual(self.runner.model_calls[0][2:], (spec.known_ids, (spec.known_sessions[3],)))
        self.assertEqual(self.result.summary.allocation.known_calibration_session, spec.known_sessions[3])

    def test_all_eight_unknowns_both_sessions_calibrate_and_never_fit(self):
        spec = self.data.protocol.specification
        self.assertEqual(self.runner.model_calls[1][2:], (spec.unknown_ids, spec.unknown_sessions))
        self.assertEqual(len(spec.unknown_ids), 8)
        self.assertFalse(set(self.runner.events[0][2]) & set(spec.unknown_ids))
        self.assertEqual(len(self.runner.model_calls), 2)  # No outer/final evaluation.

    def test_catalog_derived_counts_class_order_and_trailing_discard_counts(self):
        s = self.result.summary
        self.assertEqual(s.allocation.samples_per_condition, 8)
        self.assertEqual(s.known_fit_samples, 2 * 3 * 13 * 8)
        self.assertEqual(s.class_fit_sample_counts, (3 * 13 * 8,) * 2)
        self.assertEqual(s.known_calibration.source_samples, 2 * 13 * 8)
        self.assertEqual(s.unknown_calibration.source_samples, 8 * 2 * 13 * 8)
        self.assertEqual((s.known_calibration.windows, s.unknown_calibration.windows), (26, 208))
        self.assertEqual((s.known_calibration.discarded_samples, s.unknown_calibration.discarded_samples), (26, 208))
        self.assertEqual(self.result.model.class_order, self.data.protocol.specification.known_ids)

    def test_calibration_uses_window_7_medians_and_existing_deterministic_rule(self):
        known, correct, unknown = self.calibration_inputs[0]
        self.assertEqual((len(known), len(unknown)), (26, 208))
        replay = calibrate_threshold(known, correct, unknown, allow_negative=True)
        self.assertEqual(replay.threshold, self.result.model.threshold)
        # A deliberately bad trailing frame (index 7) is discarded; the original
        # temporal vote and median use the first seven source observations.
        self.assertTrue(correct.all())
        self.assertIn("higher threshold", self.config.threshold_policy.enumeration)

    def test_exact_same_model_no_refit_and_all_component_fingerprints_unchanged(self):
        s = self.result.summary
        self.assertEqual(self.runner.fit_count, 1)
        self.assertEqual(len({call[1] for call in self.runner.model_calls}), 1)
        self.assertEqual(s.fit_generation_before_calibration, 1)
        self.assertEqual(s.fit_generation_after_calibration, 1)
        self.assertEqual(s.fingerprints_before_calibration, s.fingerprints_after_calibration)
        self.assertEqual(s.threshold_transport, "none_same_model_calibration")
        marker = next(i for i, e in enumerate(self.runner.events) if e[0] == "calibrate")
        self.assertNotIn("fit", [e[0] for e in self.runner.events[marker + 1:]])

    def test_diagnostic_eer_cannot_replace_the_operational_threshold(self):
        def calibration(*args, **kwargs):
            return replace(calibrate_threshold(*args, **kwargs), threshold=100.)
        with patch.object(training, "calibrate_threshold", calibration):
            result = training.TrainV3DeployableModel(RecordingDeployableRunner()).train(self.data, self.config, self.provenance)
        self.assertEqual(result.model.threshold, 100.)
        self.assertNotEqual(result.summary.calibration_diagnostics.equal_error_point.threshold, 100.)

    def test_nonfinite_threshold_or_model_mutation_during_calibration_fails(self):
        with patch.object(training, "calibrate_threshold", side_effect=lambda *a, **k: replace(calibrate_threshold(*a, **k), threshold=float("inf"))):
            with self.assertRaises(RejectionError):
                training.TrainV3DeployableModel(RecordingDeployableRunner()).train(self.data, self.config, self.provenance)
        for defect in ("mutation", "refit"):
            class BadRunner(RecordingDeployableRunner):
                def infer(self, model, population):
                    result = super().infer(model, population)
                    if defect == "mutation": model.offset += 1.
                    else: self.fit_count += 1
                    return result
            with self.subTest(defect=defect), self.assertRaises(RejectionError):
                training.TrainV3DeployableModel(BadRunner()).train(self.data, self.config, self.provenance)

    def test_wrong_representation_width_nonfinite_labels_or_missing_samples_rejected(self):
        with self.assertRaises(RejectionError):
            training.deployment_allocation(replace(self.data, representation="all_xyz_v1"), self.config)
        for change in (dict(X=self.data.X[:, :-1]), dict(X=np.full_like(self.data.X, np.nan)),
                       dict(samples=self.data.samples[:-1], X=self.data.X[:-1], y=self.data.y[:-1])):
            with self.assertRaises(RejectionError): replace(self.data, **change)
        labels = self.data.y.copy(); labels[0] = -1
        with self.assertRaises(RejectionError): replace(self.data, y=labels)

    def test_unequal_group_counts_wrong_identity_provenance_and_mutable_config_rejected(self):
        # Move one observation from the first condition to the next, preserving
        # the aggregate catalog count and otherwise-valid contiguous metadata.
        samples = list(self.data.samples)
        tail = replace(samples[15], sample_index=8, detector_timestamp_ms=samples[15].detector_timestamp_ms + 250)
        order = [*range(7), *range(8, 16), 7, *range(16, len(samples))]
        changed = replace(self.data, X=self.data.X[order], y=self.data.y[order],
            samples=tuple([*samples[:7], *samples[8:16], tail, *samples[16:]]))
        with self.assertRaisesRegex(RejectionError, "per-person/session/condition counts"):
            training.deployment_allocation(changed, self.config)
        with self.assertRaises(RejectionError):
            training.deployment_allocation(self.data, replace(self.config, protocol_sha256="0" * 64))
        with self.assertRaises(RejectionError):
            training.deployment_allocation(self.data, replace(self.config, pipeline=replace(self.config.pipeline, selected_features=512)))
        with self.assertRaises(RejectionError):
            training.TrainV3DeployableModel(RecordingDeployableRunner()).train(self.data, self.config, replace(self.provenance, payload_sha256="0" * 64))

    def test_state_or_fit_generation_changed_after_threshold_selection_is_rejected(self):
        for change in ("state", "generation"):
            runner = RecordingDeployableRunner()
            fitted = []
            original_fit = runner.fit
            def fit(*args, **kwargs):
                model = original_fit(*args, **kwargs); fitted.append(model); return model
            runner.fit = fit
            def calibration(*args, **kwargs):
                value = calibrate_threshold(*args, **kwargs)
                if change == "state": fitted[0].offset += 1.
                else: runner.fit_count += 1
                return value
            with patch.object(training, "calibrate_threshold", calibration), self.assertRaises(RejectionError):
                training.TrainV3DeployableModel(runner).train(self.data, self.config, self.provenance)

    def test_aggregate_calibration_metadata_deterministic_and_never_generalization(self):
        other = training.TrainV3DeployableModel(RecordingDeployableRunner()).train(self.data, self.config, self.provenance)
        self.assertEqual(self.result.summary, other.summary)
        self.assertIn("CALIBRATION METRICS", other.summary.metrics_scope)
        self.assertIn("not independent", other.summary.metrics_scope)
        self.assertIn("V2 remains", other.summary.limitation)

    def test_value_fingerprints_ignore_array_aliases_but_detect_changed_values(self):
        array = np.arange(12, dtype=float).reshape(3, 4)
        first = dict(a=array, b=array)
        second = dict(a=array.copy(), b=np.asfortranarray(array))
        self.assertEqual(fitted_value_sha256(first), fitted_value_sha256(second))
        second["a"][0, 0] = 100.
        self.assertNotEqual(fitted_value_sha256(first), fitted_value_sha256(second))
        self.assertEqual(fitted_value_sha256(np.empty((0,))), fitted_value_sha256(np.empty((0,))))

    def test_real_components_fit_once_on_all_and_only_authorized_known_rows(self):
        from collections import Counter
        from contextlib import ExitStack
        import warnings
        from sklearn.feature_selection import SelectKBest
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        from face_recognition.infrastructure.classifiers.sklearn_v3_deployable import SklearnV3DeployableRunner
        from face_recognition.infrastructure.features.mahalanobis import PredictedClassMahalanobisScorer
        expected = self.data.known(self.data.protocol.specification.known_sessions[:3])
        active, counts = {}, Counter()
        class SpyRunner(SklearnV3DeployableRunner):
            def fit(self, population, identities, methods):
                np.testing.assert_array_equal(population.X, expected.X)
                active["population"] = population
                try: return super().fit(population, identities, methods)
                finally: active.clear()
        originals = {cls: cls.fit for cls in (SelectKBest, StandardScaler, SVC, PredictedClassMahalanobisScorer)}
        def spy(cls):
            def fit(component, X, y=None, *args, **kwargs):
                population = active["population"]
                np.testing.assert_array_equal(y, expected.y)
                self.assertEqual(X.shape, (len(expected.y), 1434 if cls is SelectKBest else 448))
                self.assertTrue(np.all(population.y >= 0))
                counts[cls] += 1
                if cls is SVC:
                    self.assertEqual((component.kernel, component.C, component.gamma, component.probability), ("rbf", 4., .001, False))
                return originals[cls](component, X, y, *args, **kwargs)
            return fit
        with ExitStack() as stack, warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning); warnings.simplefilter("ignore", RuntimeWarning)
            for cls in originals: stack.enter_context(patch.object(cls, "fit", spy(cls)))
            runner = SpyRunner(self.config)
            result = training.TrainV3DeployableModel(runner).train(self.data, self.config, self.provenance)
        self.assertEqual(runner.fit_count, 1)
        self.assertTrue(all(counts[cls] == 1 for cls in originals))
        self.assertEqual(result.model.fitted.pipeline.named_steps["anova"].k, 448)
        self.assertEqual(result.model.fitted.mahalanobis.fit_samples, len(expected.y))
