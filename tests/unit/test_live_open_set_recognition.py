from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from face_recognition.application.live_open_set_recognition import LiveOpenSetRecognition, LiveTemporalRecognition
from face_recognition.domain.models.landmark_result import DetectorSnapshot
from face_recognition.domain.models.rejection import RejectionError
from face_recognition.infrastructure.features.landmark_feature_extractor import LandmarkFeatureExtractor
from face_recognition.infrastructure.features.landmark_normalizer import LandmarkNormalizer
from face_recognition.infrastructure.features.pose_landmark_normalizer import PoseLandmarkNormalizer
from tests.live_open_set_fakes import CLASSES, Clock, face, loaded_recognizer, observation, snapshot


class LiveTemporalRecognitionTests(unittest.TestCase):
    def setUp(self):
        self.recognizer, self.runner = loaded_recognizer()
        self.state = LiveTemporalRecognition(self.recognizer)

    def feed(self, rows, start=100.):
        for i, row in enumerate(rows): self.state.observe(row, face(), start + i * .1)
        return self.state.view()

    def test_exactly_seven_observations_required_before_any_identity_decision(self):
        self.assertEqual(self.state.view().label, "NO FACE")
        for index in range(6):
            self.state.observe(observation(), face(), 100 + .1 * index)
            view = self.state.view()
            self.assertEqual(view.occupancy, index + 1)
            self.assertEqual(view.label, "COLLECTING")
            self.assertIsNone(view.last_score)
        self.state.observe(observation(), face(), 100.6)
        self.assertEqual(self.state.view().label, CLASSES[0])
        self.assertEqual(self.state.view().occupancy, 0)

    def test_nonoverlap_reset_and_last_completed_decision_remains_while_next_fills(self):
        first = self.feed([observation(0, -10.)] * 7)
        second = self.feed([observation(1, -30.)] * 6, start=100.7)
        self.assertEqual(second.label, first.label)
        self.assertEqual(second.last_score, -10.)
        self.assertEqual(second.frame_candidate, CLASSES[1])
        self.assertEqual(second.occupancy, 6)
        self.state.observe(observation(1, -30.), face(), 101.3)
        self.assertEqual(self.state.view().label, "UNKNOWN")
        self.assertEqual(self.state.view().last_score, -30.)
        self.assertEqual(self.state.view().occupancy, 0)

    def test_plurality_uses_all_frame_decision_sums_and_final_class_order_tie(self):
        indices = (0, 0, 0, 1, 1, 1, 2)
        for decisions, expected in (((0., 2., 0.), CLASSES[1]), ((0., 0., 0.), CLASSES[0])):
            state = LiveTemporalRecognition(self.recognizer)
            for i, index in enumerate(indices): state.observe(observation(index, decisions=decisions), face(), 100 + .1 * i)
            self.assertEqual(state.view().label, expected)

    def test_median_equality_known_and_below_threshold_unknown_suppresses_candidate(self):
        known = self.feed([observation(1, v) for v in (-100., -20., -19., -18., -17., -16., 0.)])
        self.assertEqual(known.last_score, -18.)
        self.assertEqual(known.label, CLASSES[1])
        state = LiveTemporalRecognition(self.recognizer)
        for i in range(7): state.observe(observation(1, -19.), face(), 100 + .1 * i)
        self.assertEqual(state.view().label, "UNKNOWN")
        self.assertEqual(state.view().temporal_candidate, CLASSES[1])
        self.assertIsNone(self.recognizer.decide_observations((observation(1, -19.),) * 7).decision.identity)

    def test_no_face_immediately_clears_partial_and_short_grace_retains_completed(self):
        self.feed([observation()] * 7)
        self.feed([observation()] * 4, start=100.7)
        self.state.no_face(101.1)
        self.assertEqual(self.state.view().occupancy, 0)
        self.assertEqual(self.state.view().label, CLASSES[0])
        self.assertIsNone(self.state.view().face)
        self.state.expire(103.1)
        self.assertEqual(self.state.view().label, "NO FACE")
        self.assertIsNone(self.state.view().last_score)
        self.assertIsNone(self.state.view().temporal_candidate)

    def test_gap_before_new_appearance_resets_partial_even_without_empty_callback(self):
        self.feed([observation()] * 4)
        self.state.observe(observation(1), face(), 103.)
        self.assertEqual(self.state.view().occupancy, 1)
        self.assertEqual(self.state.view().label, "COLLECTING")


class LiveRecognitionPipelineTests(unittest.TestCase):
    def make_pipeline(self, observations=(), delay=0.):
        clock = Clock()
        recognizer, runner = loaded_recognizer(observations, clock=clock, delay=delay)
        pipeline = LiveOpenSetRecognition(recognizer, PoseLandmarkNormalizer(640, 360), LandmarkFeatureExtractor(), clock=clock)
        return clock, pipeline, runner

    def test_pose_v2_is_used_and_model_receives_1434_float32_features(self):
        clock, pipeline, runner = self.make_pipeline()
        pipeline.process(snapshot(clock))
        expected = np.array(LandmarkFeatureExtractor().extract(PoseLandmarkNormalizer(640, 360).normalize(face())).values, dtype=np.float32)
        baseline = np.array(LandmarkFeatureExtractor().extract(LandmarkNormalizer(640, 360).normalize(face())).values, dtype=np.float32)
        np.testing.assert_array_equal(runner.inputs[0][0], expected)
        self.assertEqual(runner.inputs[0].shape, (1, 1434))
        self.assertEqual(runner.inputs[0].dtype, np.float32)
        self.assertFalse(np.allclose(expected, baseline))

    def test_repeated_busy_camera_snapshots_empty_results_and_no_face_never_advance(self):
        clock, pipeline, runner = self.make_pipeline()
        first = snapshot(clock)
        self.assertEqual(pipeline.process(first).occupancy, 1)
        for _ in range(5): self.assertEqual(pipeline.process(first).occupancy, 1)
        self.assertEqual(pipeline.process(DetectorSnapshot()).occupancy, 1)
        clock.advance(.1)
        self.assertEqual(pipeline.process(snapshot(clock, faces=())).occupancy, 0)
        self.assertEqual(runner.infer_calls, 1)
        self.assertEqual(pipeline.metrics().completed_face_inferences, 1)

    def test_first_returned_face_only_and_invalid_geometry_resets_without_unknown(self):
        clock, pipeline, runner = self.make_pipeline()
        first, second = face(), face()
        with patch.object(pipeline.normalizer, "normalize", wraps=pipeline.normalizer.normalize) as normalizer:
            pipeline.process(snapshot(clock, faces=(first, second)))
            normalizer.assert_called_once_with(first)
        clock.advance(.1)
        with patch.object(pipeline.normalizer, "normalize", side_effect=ValueError("degenerate")):
            view = pipeline.process(snapshot(clock))
        self.assertEqual(view.occupancy, 0)
        self.assertEqual(view.label, "NO FACE")
        self.assertEqual(runner.infer_calls, 1)

    def test_wrong_feature_width_rejected_before_model_inference(self):
        clock, pipeline, runner = self.make_pipeline()
        from types import SimpleNamespace
        with patch.object(pipeline.extractor, "extract", return_value=SimpleNamespace(values=(0.,) * 1431)):
            with self.assertRaises(RejectionError): pipeline.process(snapshot(clock))
        self.assertEqual(runner.infer_calls, 0)

    def test_inference_fps_counts_consumed_v3_successes_not_mp_callbacks_or_camera_frames(self):
        clock, pipeline, runner = self.make_pipeline()
        first = snapshot(clock, completed=200)
        pipeline.process(first)
        for _ in range(30): pipeline.process(first)
        clock.advance(.5)
        pipeline.process(snapshot(clock, faces=(), completed=201))
        clock.advance(.5)
        metrics = pipeline.metrics()
        self.assertEqual(metrics.completed_face_inferences, 1)
        self.assertAlmostEqual(metrics.inference_fps, 1.)

    def test_latency_includes_submission_callback_wait_pose_v3_and_temporal_work(self):
        clock, pipeline, _ = self.make_pipeline(delay=.025)
        pipeline.process(snapshot(clock, timestamp_ms=99900))  # 100 ms since submission.
        self.assertAlmostEqual(pipeline.metrics().latency_ms, 125., places=6)

    def test_stale_unseen_result_and_long_gap_cannot_mix_appearances(self):
        clock, pipeline, runner = self.make_pipeline()
        pipeline.process(snapshot(clock))
        clock.advance(3.)
        view = pipeline.process(snapshot(clock, timestamp_ms=100100))
        self.assertEqual(view.occupancy, 0)
        self.assertEqual(view.label, "NO FACE")
        self.assertEqual(runner.infer_calls, 1)
        clock.advance(.1)
        self.assertEqual(pipeline.process(snapshot(clock)).occupancy, 1)

    def test_no_fitting_or_calibration_and_batch_api_matches_frame_observations(self):
        rows = [observation(i, s) for i, s in zip((0, 0, 0, 1, 1, 1, 2), (-19., -20., -21., -18., -17., -16., -15.))]
        single, _ = loaded_recognizer(rows)
        batch, _ = loaded_recognizer(rows)
        with patch("face_recognition.application.rejection_metrics.calibrate_threshold", side_effect=AssertionError("No calibration")):
            observed = tuple(single.infer_observation(np.zeros(1434)) for _ in range(7))
            actual = single.decide_observations(observed).decision
            self.assertEqual(actual, batch.predict_window(np.zeros((7, 1434))))

    def test_model_integrity_error_is_fatal_rather_than_treated_as_no_face(self):
        clock, pipeline, _ = self.make_pipeline()
        with patch.object(pipeline.recognizer, "infer_observation", side_effect=RejectionError("changed state")):
            with self.assertRaises(RejectionError): pipeline.process(snapshot(clock))
