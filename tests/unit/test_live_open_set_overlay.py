from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from face_recognition.application.live_open_set_recognition import LiveRecognitionMetrics, LiveTemporalRecognition
from face_recognition.application.performance import PerformanceTracker
from face_recognition.domain.models.face_landmarks import FaceLandmarks, Landmark
from face_recognition.domain.models.landmark_result import DetectorSnapshot
from face_recognition.presentation.visualization.display import DRMDisplay, OpenCVDisplay
from face_recognition.presentation.visualization.live_open_set_renderer import LiveOpenSetRenderer, face_box
from tests.live_open_set_fakes import CLASSES, Clock, face, loaded_recognizer, observation


class LiveOverlayTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        recognizer, _ = loaded_recognizer()
        self.state = LiveTemporalRecognition(recognizer)
        for i in range(7): self.state.observe(observation(1, -30.), face(), 100 + .1 * i)
        self.state.observe(observation(2, -1.), face(), 100.7)
        self.metrics = replace(PerformanceTracker(clock=self.clock).snapshot(DetectorSnapshot()), capture_fps=30., completed_inference_frames=20)
        self.live = LiveRecognitionMetrics(7.4, 8, 121.)

    def test_normal_unknown_overlay_hides_both_frame_and_temporal_candidates(self):
        renderer = LiveOpenSetRenderer()
        renderer.update(self.state.view(), self.live)
        lines = renderer.metric_lines(self.metrics, "opencv")
        text = "\n".join(lines)
        self.assertEqual(lines[0], "UNKNOWN")
        self.assertNotIn(CLASSES[1], text)
        self.assertNotIn(CLASSES[2], text)
        self.assertIn("Last score (completed window): -30.00", text)
        self.assertIn("Threshold: -18.00", text)
        self.assertIn("Window: 1/7", text)
        self.assertIn("Inference FPS: 7.4", text)
        self.assertIn("121.0 ms", text)
        self.assertIn("Camera FPS (reads): 30.0", text)
        self.assertNotIn("%", text)
        self.assertNotIn("probability", text.lower())
        self.assertNotIn("confidence", text.lower())

    def test_debug_explicitly_separates_frame_score_candidate_and_last_temporal_decision(self):
        renderer = LiveOpenSetRenderer(debug=True)
        renderer.update(self.state.view(), self.live)
        text = "\n".join(renderer.metric_lines(self.metrics, "drm"))
        self.assertIn("Frame candidate: known_c / score: -1.00", text)
        self.assertIn("Temporal candidate: known_b", text)
        self.assertIn("Temporal decision: UNKNOWN", text)
        self.assertIn("MP callbacks: 20", text)
        self.assertIn("Last score (completed window): -30.00", text)

    def test_box_is_padded_and_clamped_and_nonfinite_geometry_is_ignored(self):
        bounds = FaceLandmarks((Landmark(-.2, -.3, 0.), Landmark(1.2, 1.3, 0.)))
        self.assertEqual(face_box(bounds, 100, 50), (0, 0, 99, 49))
        normal = FaceLandmarks((Landmark(.3, .3, 0.), Landmark(.7, .7, 0.)))
        x0, y0, x1, y1 = face_box(normal, 100, 100)
        self.assertLess(x0, 30); self.assertLess(y0, 30)
        self.assertGreater(x1, 69); self.assertGreater(y1, 69)
        self.assertIsNone(face_box(FaceLandmarks((Landmark(float("nan"), 0., 0.),)), 100, 50))

    def test_expired_face_removes_box_even_if_landmarker_snapshot_still_contains_old_face(self):
        self.state.expire(103.)
        renderer = LiveOpenSetRenderer()
        renderer.update(self.state.view(), self.live)
        with patch("face_recognition.presentation.visualization.live_open_set_renderer.cv2.rectangle") as rectangle:
            output = renderer.render_overlay(640, 480, (face(),), self.metrics)
        rectangle.assert_not_called()
        self.assertEqual(output.shape, (480, 640, 4))
        self.assertEqual(renderer.metric_lines(self.metrics, "drm")[0], "NO FACE")

    def test_opencv_supports_escape_and_q_without_affecting_old_default_exit_keys(self):
        frame = SimpleNamespace(frame=np.zeros((10, 10, 3), dtype=np.uint8), faces=())
        renderer = Mock()
        for key in (ord("q"), 27):
            display = OpenCVDisplay(renderer, window_title="V3", exit_keys=(ord("q"), 27))
            with patch("face_recognition.presentation.visualization.display.cv2.imshow"), \
                    patch("face_recognition.presentation.visualization.display.cv2.waitKey", return_value=key):
                self.assertFalse(display.show(frame, self.metrics))
        self.assertEqual(OpenCVDisplay(renderer).exit_keys, (ord("q"),))

    def test_drm_refreshes_immediate_state_reset_without_new_callback_or_metrics_interval(self):
        renderer = Mock(mode="all", revision=0)
        renderer.render_overlay.return_value = np.zeros((480, 640, 4), dtype=np.uint8)
        target = Mock()
        display = DRMDisplay(target, renderer, 640, 480, clock=self.clock)
        frame = SimpleNamespace(detector_snapshot=DetectorSnapshot(), faces=())
        display.show(frame, self.metrics)
        self.clock.advance(.1)
        display.show(frame, self.metrics)
        self.assertFalse(display.overlay_updated)
        renderer.revision = 1
        display.show(frame, self.metrics)
        self.assertTrue(display.overlay_updated)
        self.assertEqual(target.set_overlay.call_count, 2)
        display.close()
