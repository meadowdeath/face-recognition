import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from face_recognition.config.settings import Settings
from face_recognition.domain.models.landmark_result import DetectorSnapshot
from face_recognition.infrastructure.camera import picamera2_camera as pi_adapter
from face_recognition.presentation.cli import recognize as cli


class PreviewCliTests(unittest.TestCase):
    def test_q_exits_and_releases_all_resources(self) -> None:
        camera = Mock()
        detector = Mock()
        camera.read.return_value = np.zeros((4, 4, 3), dtype=np.uint8)
        detector.snapshot.return_value = DetectorSnapshot()
        renderer = Mock()
        with patch.object(cli, "OpenCVCamera", return_value=camera), \
             patch.object(cli, "MediaPipeFaceLandmarker", return_value=detector), \
             patch.object(cli.cv2, "imshow") as show, \
             patch.object(cli.cv2, "waitKey", return_value=ord("q")), \
             patch.object(cli.cv2, "destroyAllWindows") as destroy:
            cli.run_preview(Settings(), renderer)
        detector.submit.assert_called_once()
        show.assert_called_once()
        detector.close.assert_called_once()
        camera.close.assert_called_once()
        destroy.assert_called_once()

    def test_each_backend_uses_the_shared_loop_and_configured_dimensions(self) -> None:
        settings = Settings(camera_width=320, camera_height=240)
        for backend in ("opencv", "picamera2"):
            with self.subTest(backend=backend):
                camera = Mock()
                detector = Mock()
                frame = np.zeros((240, 320, 3), dtype=np.uint8)
                camera.read.return_value = frame
                detector.snapshot.return_value = DetectorSnapshot()
                with patch.object(cli, "OpenCVCamera", return_value=camera) as opencv, \
                     patch.object(cli, "Picamera2Camera", return_value=camera) as picamera2, \
                     patch.object(cli, "MediaPipeFaceLandmarker", return_value=detector), \
                     patch.object(cli.cv2, "imshow"), \
                     patch.object(cli.cv2, "waitKey", return_value=ord("q")), \
                     patch.object(cli.cv2, "destroyAllWindows"):
                    cli.run_preview(settings, Mock(), camera_backend=backend)
                if backend == "opencv":
                    opencv.assert_called_once_with(settings.camera_index, 320, 240, settings.camera_fps)
                    picamera2.assert_not_called()
                else:
                    picamera2.assert_called_once_with(320, 240)
                    opencv.assert_not_called()
                detector.submit.assert_called_once_with(frame)
                camera.close.assert_called_once()
                detector.close.assert_called_once()

    def test_camera_initialization_failure_closes_detector(self) -> None:
        for backend, adapter in (("opencv", "OpenCVCamera"), ("picamera2", "Picamera2Camera")):
            with self.subTest(backend=backend), \
                 patch.object(cli, adapter, side_effect=RuntimeError("camera init failed")), \
                 patch.object(cli, "MediaPipeFaceLandmarker") as factory, \
                 patch.object(cli.cv2, "imshow") as show:
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
                     patch.dict("sys.modules", {"picamera2": fake_module}):
                    with self.assertRaisesRegex(RuntimeError, "Pi init failed"):
                        pi_adapter.Picamera2Camera(320, 240)
                native.close.assert_called_once()

    def test_picamera2_close_releases_camera_even_when_stop_fails(self) -> None:
        native = Mock()
        fake_module = SimpleNamespace(Picamera2=Mock(return_value=native))
        with patch.object(pi_adapter.sys, "platform", "linux"), \
             patch.dict("sys.modules", {"picamera2": fake_module}):
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
             patch.object(cli.cv2, "destroyAllWindows") as destroy:
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
