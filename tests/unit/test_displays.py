import unittest
from unittest.mock import Mock, patch

from face_recognition.application.performance import PerformanceTracker
from face_recognition.application.recognize_face import LandmarkFrame
from face_recognition.domain.models.landmark_result import DetectorSnapshot
from face_recognition.presentation.visualization.display import DRMDisplay, NoDisplay
from face_recognition.presentation.visualization.frame_renderer import FrameRenderer


class DisplayTests(unittest.TestCase):
    def setUp(self) -> None:
        # Not an image: DRM/no-display must not inspect or copy the camera frame.
        self.frame = LandmarkFrame(object(), DetectorSnapshot())
        self.metrics = PerformanceTracker().snapshot(DetectorSnapshot())

    def test_drm_builds_an_overlay_without_accessing_camera_pixels(self) -> None:
        target = Mock()
        display = DRMDisplay(target, FrameRenderer(), 200, 160)
        self.assertTrue(display.show(self.frame, self.metrics))
        overlay = target.set_overlay.call_args.args[0]
        self.assertEqual(overlay.shape, (160, 200, 4))
        self.assertEqual(overlay[100, 0, 3], 0)
        display.close()
        display.close()
        self.assertEqual(target.set_overlay.call_count, 2)
        self.assertIsNone(target.set_overlay.call_args.args[0])
        target.stop_preview.assert_called_once()

    def test_no_display_reports_periodically_without_rendering(self) -> None:
        now = [0.0]
        renderer = FrameRenderer()
        display = NoDisplay(renderer, 1.0, clock=lambda: now[0])
        with patch.object(renderer, "render") as render, \
             patch.object(renderer, "render_overlay") as overlay, patch("builtins.print") as output:
            self.assertTrue(display.show(self.frame, self.metrics))
            output.assert_not_called()
            now[0] = 1.0
            self.assertTrue(display.show(self.frame, self.metrics))
            display.show(self.frame, self.metrics)
            output.assert_called_once()
            report = output.call_args.args[0]
            self.assertIn("Capture:", report)
            self.assertIn("Inference:", report)
            self.assertIn("Submitted:", report)
            self.assertIn("Completed:", report)
            self.assertNotIn("Display:", report)
            self.assertNotIn("Overlay updates:", report)
            render.assert_not_called()
            overlay.assert_not_called()
