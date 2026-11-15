"""Hardware-free hot-path/inference equivalence and opt-in coarse timing tests."""

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from face_recognition.application.live_open_set_recognition import LiveOpenSetRecognition
from face_recognition.application.v3_deployable_recognition import V3DeployableRecognizer
from face_recognition.domain.models.rejection import RejectionError
from face_recognition.infrastructure.classifiers.sklearn_v3_same_model import SklearnV3SameModelRunner
from face_recognition.infrastructure.features.landmark_feature_extractor import LandmarkFeatureExtractor
from face_recognition.infrastructure.features.pose_landmark_normalizer import PoseLandmarkNormalizer
from tests.live_open_set_fakes import Clock, InferenceOnlyRunner, loaded_recognizer, observation, snapshot


class DeepCheckedRecognizer(V3DeployableRecognizer):
    """Reference: the old deep-check-before/after-each-inference policy."""
    def _verify(self):
        super()._verify()
        if self._runner.fingerprints(self._model.fitted) != self._expected:
            raise RejectionError("Reference fitted state changed")


class V3LiveIntegrityPerformanceTests(unittest.TestCase):
    def test_startup_still_compares_expected_fingerprints(self):
        with patch.object(InferenceOnlyRunner, "fingerprints", autospec=True, side_effect=lambda runner, model: model.version) as check:
            service, runner = loaded_recognizer()
            self.assertEqual(check.call_count, 1)
            self.assertEqual(runner.infer_calls, 0)
            with self.assertRaises(RejectionError): V3DeployableRecognizer(service._model, runner, 999)

    def test_many_observations_and_temporal_decisions_do_not_hash_or_serialize(self):
        service, runner = loaded_recognizer()
        with patch.object(runner, "fingerprints", side_effect=AssertionError("No per-frame fingerprints")), \
                patch("face_recognition.infrastructure.classifiers.sklearn_v3_deployable.fitted_v3_fingerprints", side_effect=AssertionError("No deep hash")), \
                patch("face_recognition.infrastructure.classifiers.v3_fitted_integrity.fitted_value_sha256", side_effect=AssertionError("No value hashing")), \
                patch("pickle.dump", side_effect=AssertionError("No model serialization")):
            for _ in range(3):
                rows = tuple(service.infer_observation(np.zeros(1434)) for _ in range(7))
                self.assertEqual(service.decide_observations(rows).decision.identity, "known_a")
            service.predict_window(np.zeros((7, 1434)))
        self.assertEqual(runner.infer_calls, 22)

    def test_fit_generation_mutation_remains_fatal(self):
        service, runner = loaded_recognizer()
        runner.fit_count += 1
        with self.assertRaises(RejectionError): service.infer_observation(np.zeros(1434))
        self.assertEqual(runner.infer_calls, 0)

    def test_component_replacement_detected_using_existing_runner_identity_guard(self):
        class Pipeline:
            def __init__(self): self.named_steps = {name: object() for name in ("anova", "scaler", "classifier")}
            def __getitem__(self, index): return self.named_steps["classifier"]
        for component in ("anova", "scaler", "classifier", "mahalanobis", "pipeline"):
            base, runner = loaded_recognizer()
            fitted = base._model.fitted
            fitted.pipeline, fitted.mahalanobis = Pipeline(), object()
            runner.model_identity = SklearnV3SameModelRunner.model_identity
            service = V3DeployableRecognizer(base._model, runner, 1)
            if component in fitted.pipeline.named_steps: fitted.pipeline.named_steps[component] = object()
            else: setattr(fitted, component, Pipeline() if component == "pipeline" else object())
            with self.subTest(component=component), self.assertRaises(RejectionError):
                service.infer_observation(np.zeros(1434))

    def test_model_runner_fitted_reference_and_frozen_scalar_changes_rejected(self):
        for target in ("model", "runner", "fitted", "threshold", "representation", "input_width", "window"):
            service, _ = loaded_recognizer()
            if target == "model": service._model = replace(service._model)
            elif target == "runner": service._runner = InferenceOnlyRunner()
            elif target == "fitted": object.__setattr__(service._model, "fitted", SimpleNamespace(identity_count=3, version=1))
            elif target == "threshold": object.__setattr__(service._model, "threshold", float("nan"))
            elif target == "representation": service._model.frozen_configuration.pipeline.representation = "all_xyz_v1"
            elif target == "input_width": service._model.frozen_configuration.pipeline.input_features = 1431
            else: service._model.frozen_configuration.temporal.window_size = 5
            with self.subTest(target=target), self.assertRaises(RejectionError): service.infer_observation(np.zeros(1434))

    def test_explicit_full_verification_detects_in_place_value_changes(self):
        service, runner = loaded_recognizer()
        service._model.fitted.version += 1
        with self.assertRaises(RejectionError): service.verify_integrity()
        self.assertEqual(runner.infer_calls, 0)

    def test_candidate_scores_and_window_decision_equal_deep_checked_reference(self):
        rows = [observation(i, s, decisions=(0., 2., 0.)) for i, s in
                zip((0, 0, 0, 1, 1, 1, 2), (-100., -20., -19., -18., -17., -16., 0.))]
        fast, _ = loaded_recognizer(rows)
        base, runner = loaded_recognizer(rows)
        reference = DeepCheckedRecognizer(base._model, runner, 1)
        features = np.random.default_rng(73).normal(size=(7, 1434)).astype(np.float32)
        actual = tuple(fast.infer_observation(row) for row in features)
        expected = tuple(reference.infer_observation(row) for row in features)
        self.assertEqual(actual, expected)  # All candidate IDs, indices, decision scores and knownness.
        result = fast.decide_observations(actual)
        self.assertEqual(result, reference.decide_observations(expected))
        self.assertEqual(result.candidate_identity, "known_b")
        self.assertEqual(result.decision.knownness_score, -18.)
        self.assertEqual(result.decision.identity, "known_b")  # Equality accepts.

    def test_unknown_suppression_unchanged_and_no_training_api_or_calls(self):
        service, runner = loaded_recognizer([observation(1, -24.)] * 7)
        with patch.object(runner, "fit", side_effect=AssertionError("No fitting")), \
                patch("face_recognition.application.rejection_metrics.calibrate_threshold", side_effect=AssertionError("No calibration")):
            result = service.predict_window(np.zeros((7, 1434)))
        self.assertEqual(result.decision.value, "UNKNOWN")
        self.assertIsNone(result.identity)
        self.assertEqual(result.threshold, -18.)
        self.assertFalse(hasattr(service, "fit"))
        self.assertFalse(hasattr(service, "calibrate"))

    def test_debug_timings_are_finite_nonnegative_and_do_not_change_callback_latency(self):
        clock = Clock()
        recognizer, _ = loaded_recognizer(clock=clock, delay=.025)
        normalizer = PoseLandmarkNormalizer(640, 360)
        extractor = LandmarkFeatureExtractor()
        pipeline = LiveOpenSetRecognition(recognizer, normalizer, extractor, clock=clock, profile=True)
        normalize, extract = normalizer.normalize, extractor.extract
        def timed_normalize(face): clock.advance(.006); return normalize(face)
        def timed_extract(face): clock.advance(.004); return extract(face)
        source = snapshot(clock)
        with patch.object(normalizer, "normalize", side_effect=timed_normalize), patch.object(extractor, "extract", side_effect=timed_extract):
            pipeline.process(source)
        metrics = pipeline.metrics()
        self.assertAlmostEqual(metrics.pose_features_ms, 10., places=6)
        self.assertAlmostEqual(metrics.v3_observation_ms, 25., places=6)
        self.assertAlmostEqual(metrics.latency_ms, 35., places=6)
        self.assertEqual(source.result.latency_ms, 0.)
        self.assertTrue(all(np.isfinite(v) and v >= 0 for v in (metrics.pose_features_ms, metrics.v3_observation_ms)))

    def test_profiling_is_opt_in_with_no_extra_clock_reads_in_normal_path(self):
        clock = Mock(side_effect=[100.] * 10)
        recognizer, _ = loaded_recognizer()
        pipeline = LiveOpenSetRecognition(recognizer, PoseLandmarkNormalizer(640, 360), LandmarkFeatureExtractor(), clock=clock)
        before = clock.call_count
        pipeline.process(snapshot(Clock()))
        self.assertEqual(clock.call_count - before, 3)  # Existing now/completed/finished only.
        metrics = pipeline.metrics()
        self.assertIsNone(metrics.pose_features_ms)
        self.assertIsNone(metrics.v3_observation_ms)
