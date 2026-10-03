import unittest
from unittest.mock import Mock, patch

from face_recognition.application.performance import PerformanceTracker
from face_recognition.application.recognize_face import LandmarkFrame
from face_recognition.domain.models.face_landmarks import FaceLandmarks, Landmark
from face_recognition.domain.models.landmark_result import DetectorSnapshot, LandmarkResult
from face_recognition.presentation.visualization.display import DRMDisplay, NoDisplay, OpenCVDisplay
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
            self.assertIn("Skipped Busy:", report)
            self.assertIn("Captured:", report)
            self.assertIn("Result age:", report)
            self.assertNotIn("Display:", report)
            self.assertNotIn("Overlay updates:", report)
            render.assert_not_called()
            overlay.assert_not_called()

    def test_drm_updates_only_for_new_results_or_metrics_in_landmark_modes(self) -> None:
        face = FaceLandmarks((Landmark(0.5, 0.5, 0.0),))
        first = LandmarkFrame(object(), DetectorSnapshot(LandmarkResult((face,), 100, 20)))
        second = LandmarkFrame(object(), DetectorSnapshot(LandmarkResult((face,), 200, 20)))
        no_face = LandmarkFrame(object(), DetectorSnapshot(LandmarkResult((), 300, 20)))
        for mode, indices in (("all", ()), ("selected", (0,))):
            with self.subTest(mode=mode):
                now = [0.0]
                target = Mock()
                renderer = FrameRenderer(mode, indices)
                display = DRMDisplay(target, renderer, 848, 480, clock=lambda: now[0])
                with patch.object(renderer, "render_overlay", wraps=renderer.render_overlay) as render:
                    display.show(first, self.metrics)
                    self.assertTrue(display.overlay_updated)
                    for _ in range(5):
                        now[0] = 0.1
                        display.show(first, self.metrics)
                        self.assertFalse(display.overlay_updated)
                    render.assert_called_once()
                    target.set_overlay.assert_called_once()
                    now[0] = 0.9
                    display.show(second, self.metrics)
                    self.assertTrue(display.overlay_updated)
                    self.assertEqual(target.set_overlay.call_count, 2)
                    # New-result redraws must not postpone the periodic timer.
                    now[0] = 1.0
                    display.show(second, self.metrics)
                    self.assertTrue(display.overlay_updated)
                    self.assertEqual(target.set_overlay.call_count, 3)
                    self.assertEqual(render.call_args.args[:3], (848, 480, (face,)))
                    self.assertEqual(target.set_overlay.call_args.args[0].shape, (480, 848, 4))
                    now[0] = 1.1
                    display.show(no_face, self.metrics)
                    self.assertEqual(render.call_args.args[2], ())
                    self.assertEqual(target.set_overlay.call_count, 4)
                    display.show(no_face, self.metrics)
                    self.assertEqual(target.set_overlay.call_count, 4)
                display.close()
                target.set_overlay.assert_called_with(None)
                target.stop_preview.assert_called_once()

    def test_drm_none_updates_only_for_metrics_even_with_new_results(self) -> None:
        now = [0.0]
        target = Mock()
        renderer = FrameRenderer("none")
        display = DRMDisplay(target, renderer, 848, 480, interval_seconds=2.0, clock=lambda: now[0])
        with patch.object(renderer, "render_overlay", wraps=renderer.render_overlay) as render:
            display.show(self.frame, self.metrics)  # Initial pending metrics.
            for timestamp in (100, 200, 300):
                now[0] += 0.5
                frame = LandmarkFrame(object(), DetectorSnapshot(LandmarkResult((), timestamp, 20)))
                display.show(frame, self.metrics)
                self.assertFalse(display.overlay_updated)
            render.assert_called_once()
            target.set_overlay.assert_called_once()
            now[0] = 2.0
            display.show(frame, self.metrics)
            self.assertTrue(display.overlay_updated)
            self.assertEqual(render.call_count, 2)
            now[0] = 4.0
            display.show(frame, self.metrics)  # Refresh without any new result.
            self.assertEqual(target.set_overlay.call_count, 3)
        display.close()
        target.set_overlay.assert_called_with(None)

    def test_drm_metrics_refresh_without_any_completed_result(self) -> None:
        now = [0.0]
        target = Mock()
        display = DRMDisplay(target, FrameRenderer(), 848, 480, clock=lambda: now[0])
        display.show(self.frame, self.metrics)
        now[0] = 0.5
        display.show(self.frame, self.metrics)
        self.assertEqual(target.set_overlay.call_count, 1)
        now[0] = 1.0
        display.show(self.frame, self.metrics)
        self.assertEqual(target.set_overlay.call_count, 2)
        display.close()

    def test_drm_failed_submission_does_not_mark_an_overlay_update(self) -> None:
        target = Mock()
        target.set_overlay.side_effect = RuntimeError("overlay failed")
        display = DRMDisplay(target, FrameRenderer(), 848, 480)
        with self.assertRaisesRegex(RuntimeError, "overlay failed"):
            display.show(self.frame, self.metrics)
        self.assertFalse(display.overlay_updated)
        self.assertFalse(display._has_overlay)
        target.set_overlay.side_effect = None
        display.close()

    def test_opencv_still_renders_and_pumps_gui_on_every_frame(self) -> None:
        renderer = Mock()
        display = OpenCVDisplay(renderer)
        with patch("cv2.imshow") as show, patch("cv2.waitKey", return_value=0) as wait, \
             patch("cv2.destroyAllWindows") as destroy:
            for _ in range(3):
                self.assertTrue(display.show(self.frame, self.metrics))
            self.assertEqual(renderer.render.call_count, 3)
            self.assertEqual(show.call_count, 3)
            self.assertEqual(wait.call_count, 3)
            display.close()
            destroy.assert_called_once()
