import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from face_recognition.config.settings import Settings
from face_recognition.domain.models.landmark_result import DetectorSnapshot
from face_recognition.infrastructure.camera import picamera2_camera as pi_adapter
from face_recognition.presentation.cli import recognize as cli
from face_recognition.presentation.visualization import display as displays


class PreviewCliTests(unittest.TestCase):
    def test_drm_counts_actual_updates_not_capture_iterations(self) -> None:
        camera = Mock()
        camera.read.side_effect = [object(), object(), object(), object(), KeyboardInterrupt()]
        detector = Mock()
        detector.snapshot.return_value = DetectorSnapshot()
        renderer = Mock(mode="none")
        tracker = cli.PerformanceTracker()
        clock = Mock(side_effect=[0.0, 0.0, 0.1, 0.2, 2.0])

        def make_display(*args, **kwargs):
            return displays.DRMDisplay(*args, clock=clock, **kwargs)

        with patch.object(cli, "Picamera2Camera", return_value=camera), \
             patch.object(cli, "MediaPipeFaceLandmarker", return_value=detector), \
             patch.object(cli, "PerformanceTracker", return_value=tracker), \
             patch.object(cli, "DRMDisplay", side_effect=make_display):
            cli.run_preview(Settings(metrics_interval_seconds=2.0), renderer, "picamera2", "drm")
        metrics = tracker.snapshot(DetectorSnapshot())
        self.assertEqual(metrics.captured_frames, 4)
        self.assertEqual(metrics.overlay_updates, 2)
        self.assertEqual(metrics.displayed_frames, 0)
        self.assertEqual(renderer.render_overlay.call_count, 2)
        self.assertEqual(camera.set_overlay.call_count, 3)  # Two updates + cleanup.
        camera.set_overlay.assert_called_with(None)
        self.assertEqual(detector.submit.call_count, 4)
        camera.close.assert_called_once()
        detector.close.assert_called_once()

    def test_shared_resolution_defaults_and_overrides(self) -> None:
        for backend in ("opencv", "picamera2"):
            with self.subTest(backend=backend), patch.object(cli, "run_preview") as run:
                cli.main(["--camera", backend, "--display", "none"])
                settings = run.call_args.args[0]
                self.assertEqual((settings.display_width, settings.display_height), (848, 480))
                self.assertEqual((settings.inference_width, settings.inference_height), (480, 270))
                cli.main(["--camera", backend, "--display", "none", "--orientation", "rotate180",
                          "--display-width", "640", "--display-height", "360",
                          "--inference-width", "320", "--inference-height", "180"])
                settings = run.call_args.args[0]
                self.assertEqual(settings.camera_orientation, "rotate180")
                self.assertEqual((settings.display_width, settings.display_height), (640, 360))
                self.assertEqual((settings.inference_width, settings.inference_height), (320, 180))

    def test_invalid_dimensions_and_orientation_fail_before_opening_resources(self) -> None:
        for arguments in (["--orientation", "rotate90"], ["--display-width", "0"],
                          ["--camera", "picamera2", "--inference-width", "481"],
                          ["--camera", "picamera2", "--inference-height", "482"]):
            with self.subTest(arguments=arguments), patch.object(cli, "run_preview") as run, patch("sys.stderr"):
                with self.assertRaises(SystemExit):
                    cli.main(arguments)
                run.assert_not_called()

    def test_q_exits_and_releases_all_resources(self) -> None:
        camera = Mock()
        detector = Mock()
        camera.read.return_value = np.zeros((4, 4, 3), dtype=np.uint8)
        detector.snapshot.return_value = DetectorSnapshot()
        renderer = Mock()
        with patch.object(cli, "OpenCVCamera", return_value=camera), \
             patch.object(cli, "MediaPipeFaceLandmarker", return_value=detector), \
             patch.object(displays.cv2, "imshow") as show, \
             patch.object(displays.cv2, "waitKey", return_value=ord("q")), \
             patch.object(displays.cv2, "destroyAllWindows") as destroy, \
             patch.dict("sys.modules", {"picamera2": None, "libcamera": None}):
            cli.run_preview(Settings(), renderer)
        detector.submit.assert_called_once()
        show.assert_called_once()
        detector.close.assert_called_once()
        camera.close.assert_called_once()
        destroy.assert_called_once()

    def test_each_backend_uses_the_shared_loop_and_configured_dimensions(self) -> None:
        settings = Settings(display_width=320, display_height=240, inference_width=160, inference_height=120)
        for backend in ("opencv", "picamera2"):
            with self.subTest(backend=backend):
                camera = Mock()
                detector = Mock()
                frame = np.zeros((240, 320, 3), dtype=np.uint8)
                camera.read.return_value = frame
                detector.snapshot.return_value = DetectorSnapshot()
                with patch.object(cli, "OpenCVCamera", return_value=camera) as opencv, \
                     patch.object(cli, "Picamera2Camera", return_value=camera) as picamera2, \
                     patch.object(cli, "MediaPipeFaceLandmarker", return_value=detector) as landmarker, \
                     patch.object(displays.cv2, "imshow"), \
                     patch.object(displays.cv2, "waitKey", return_value=ord("q")), \
                     patch.object(displays.cv2, "destroyAllWindows"):
                    cli.run_preview(settings, Mock(), camera_backend=backend)
                if backend == "opencv":
                    opencv.assert_called_once_with(settings.camera_index, 320, 240, settings.camera_fps,
                                                   orientation="normal")
                    picamera2.assert_not_called()
                else:
                    picamera2.assert_called_once_with(160, 120, display_width=320, display_height=240,
                                                     orientation="normal")
                    opencv.assert_not_called()
                detector.submit.assert_called_once_with(frame)
                self.assertEqual(landmarker.call_args.kwargs["inference_size"], (160, 120))
                camera.close.assert_called_once()
                detector.close.assert_called_once()

    def test_camera_initialization_failure_closes_detector(self) -> None:
        for backend, adapter in (("opencv", "OpenCVCamera"), ("picamera2", "Picamera2Camera")):
            with self.subTest(backend=backend), \
                 patch.object(cli, adapter, side_effect=RuntimeError("camera init failed")), \
                 patch.object(cli, "MediaPipeFaceLandmarker") as factory, \
                 patch.object(displays.cv2, "imshow") as show:
                with self.assertRaisesRegex(RuntimeError, "camera init failed"):
                    cli.run_preview(Settings(), Mock(), camera_backend=backend)
                factory.return_value.close.assert_called_once()
                show.assert_not_called()

    def test_cli_selects_backend_and_defaults_to_opencv(self) -> None:
        for arguments, expected in (([], "opencv"), (["--camera", "opencv"], "opencv"),
                                    (["--camera", "picamera2"], "picamera2")):
            with self.subTest(arguments=arguments), patch.object(cli, "run_preview") as run:
                cli.main(arguments)
                self.assertEqual(run.call_args.kwargs["camera_backend"], expected)
        with patch.object(cli, "run_preview") as run, patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                cli.main(["--camera", "unsupported"])
            run.assert_not_called()

    def test_partial_picamera2_initialization_is_closed_without_hardware(self) -> None:
        for failing_operation in ("create_preview_configuration", "configure", "start"):
            with self.subTest(operation=failing_operation):
                native = Mock()
                getattr(native, failing_operation).side_effect = RuntimeError("Pi init failed")
                fake_module = SimpleNamespace(Picamera2=Mock(return_value=native))
                with patch.object(pi_adapter.sys, "platform", "linux"), \
                     patch.dict("sys.modules", {"picamera2": fake_module,
                                                "libcamera": SimpleNamespace(Transform=Mock())}):
                    with self.assertRaisesRegex(RuntimeError, "Pi init failed"):
                        pi_adapter.Picamera2Camera(320, 240)
                native.close.assert_called_once()

    def test_picamera2_close_releases_camera_even_when_stop_fails(self) -> None:
        native = Mock()
        fake_module = SimpleNamespace(Picamera2=Mock(return_value=native))
        with patch.object(pi_adapter.sys, "platform", "linux"), \
             patch.dict("sys.modules", {"picamera2": fake_module,
                                        "libcamera": SimpleNamespace(Transform=Mock())}):
            camera = pi_adapter.Picamera2Camera(320, 240)
        native.stop.side_effect = RuntimeError("stop failed")
        with self.assertRaisesRegex(RuntimeError, "stop failed"):
            camera.close()
        native.close.assert_called_once()

    def test_inference_submission_error_still_cleans_up(self) -> None:
        camera = Mock()
        detector = Mock()
        camera.read.return_value = object()
        detector.submit.side_effect = RuntimeError("submission failed")
        with patch.object(cli, "OpenCVCamera", return_value=camera), \
             patch.object(cli, "MediaPipeFaceLandmarker", return_value=detector), \
             patch.object(displays.cv2, "destroyAllWindows") as destroy:
            with self.assertRaisesRegex(RuntimeError, "submission failed"):
                cli.run_preview(Settings(), Mock())
        detector.close.assert_called_once()
        camera.close.assert_called_once()
        destroy.assert_called_once()

    def test_selected_mode_requires_explicit_indices_before_opening_camera(self) -> None:
        with patch.object(cli, "run_preview") as run, patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                cli.main(["--landmarks", "selected"])
        run.assert_not_called()
        with patch.object(cli, "run_preview") as run:
            cli.main(["--landmarks", "selected", "--indices", "1,4,1"])
        self.assertEqual(run.call_args.args[1].selected_indices, (1, 4))

    def test_cli_display_selection_and_invalid_drm_combination(self) -> None:
        for backend in ("opencv", "drm", "none"):
            with self.subTest(display=backend), patch.object(cli, "run_preview") as run:
                cli.main(["--camera", "picamera2", "--display", backend])
                self.assertEqual(run.call_args.kwargs["display_backend"], backend)
        with patch.object(cli, "run_preview") as run, patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                cli.main(["--camera", "opencv", "--display", "drm"])
            run.assert_not_called()

    def test_drm_ctrl_c_cleans_up_without_opencv_gui_calls(self) -> None:
        camera = Mock()
        detector = Mock()
        frame = np.zeros((270, 480, 3), dtype=np.uint8)
        camera.read.side_effect = [frame, KeyboardInterrupt()]
        detector.snapshot.return_value = DetectorSnapshot()
        renderer = Mock()
        overlay = np.zeros((480, 848, 4), dtype=np.uint8)
        renderer.render_overlay.return_value = overlay
        events = Mock()
        events.attach_mock(camera, "camera")
        events.attach_mock(detector, "detector")
        tracker = cli.PerformanceTracker()
        with patch.object(cli, "Picamera2Camera", return_value=camera), \
             patch.object(cli, "MediaPipeFaceLandmarker", return_value=detector), \
             patch.object(cli, "PerformanceTracker", return_value=tracker), \
             patch.object(displays.cv2, "imshow") as show, \
             patch.object(displays.cv2, "waitKey") as wait, \
             patch.object(displays.cv2, "destroyAllWindows") as destroy:
            cli.run_preview(Settings(), renderer, "picamera2", "drm")
        camera.start_drm_preview.assert_called_once_with(848, 480)
        self.assertEqual(renderer.render_overlay.call_args.args[:2], (848, 480))
        self.assertIs(camera.set_overlay.call_args_list[0].args[0], overlay)
        self.assertIsNone(camera.set_overlay.call_args_list[1].args[0])
        camera.stop_preview.assert_called_once()
        camera.close.assert_called_once()
        detector.close.assert_called_once()
        renderer.render.assert_not_called()
        show.assert_not_called()
        wait.assert_not_called()
        destroy.assert_not_called()
        metrics = tracker.snapshot(DetectorSnapshot())
        self.assertEqual(metrics.overlay_updates, 1)
        self.assertEqual(metrics.displayed_frames, 0)
        # Display-specific cleanup precedes camera and detector shutdown.
        names = [call[0] for call in events.mock_calls]
        self.assertEqual(names[names.index("camera.stop_preview") - 1], "camera.set_overlay")
        self.assertLess(names.index("camera.stop_preview"), names.index("camera.close"))

    def test_no_display_ctrl_c_skips_all_rendering_and_window_calls(self) -> None:
        camera = Mock()
        detector = Mock()
        camera.read.side_effect = [object(), KeyboardInterrupt()]
        detector.snapshot.return_value = DetectorSnapshot()
        renderer = Mock()
        with patch.object(cli, "Picamera2Camera", return_value=camera), \
             patch.object(cli, "MediaPipeFaceLandmarker", return_value=detector), \
             patch.object(displays.cv2, "imshow") as show, \
             patch.object(displays.cv2, "waitKey") as wait, \
             patch.object(displays.cv2, "destroyAllWindows") as destroy:
            cli.run_preview(Settings(), renderer, "picamera2", "none")
        detector.submit.assert_called_once()
        renderer.render.assert_not_called()
        renderer.render_overlay.assert_not_called()
        camera.start_drm_preview.assert_not_called()
        camera.set_overlay.assert_not_called()
        camera.close.assert_called_once()
        detector.close.assert_called_once()
        show.assert_not_called()
        wait.assert_not_called()
        destroy.assert_not_called()

    def test_drm_start_failure_still_closes_camera_and_detector(self) -> None:
        camera = Mock()
        detector = Mock()
        camera.start_drm_preview.side_effect = RuntimeError("DRM unavailable")
        with patch.object(cli, "Picamera2Camera", return_value=camera), \
             patch.object(cli, "MediaPipeFaceLandmarker", return_value=detector):
            with self.assertRaisesRegex(RuntimeError, "DRM unavailable"):
                cli.run_preview(Settings(), Mock(), "picamera2", "drm")
        camera.close.assert_called_once()
        detector.close.assert_called_once()

    def test_drm_overlay_cleanup_failure_still_stops_and_closes_resources(self) -> None:
        camera = Mock()
        detector = Mock()
        camera.read.side_effect = KeyboardInterrupt()
        camera.set_overlay.side_effect = RuntimeError("overlay clear failed")
        with patch.object(cli, "Picamera2Camera", return_value=camera), \
             patch.object(cli, "MediaPipeFaceLandmarker", return_value=detector):
            with self.assertRaisesRegex(RuntimeError, "overlay clear failed"):
                cli.run_preview(Settings(), Mock(), "picamera2", "drm")
        camera.set_overlay.assert_called_once_with(None)
        camera.stop_preview.assert_called_once()
        camera.close.assert_called_once()
        detector.close.assert_called_once()
