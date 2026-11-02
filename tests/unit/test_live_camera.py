import builtins
import unittest
from unittest.mock import Mock, patch

from face_recognition.infrastructure.camera.live_camera import create_live_camera
from face_recognition.infrastructure.camera.opencv_camera import OpenCVCamera


class LiveCameraTests(unittest.TestCase):
    def test_opencv_backend_never_imports_or_requires_picamera2(self):
        original = builtins.__import__
        def guard(name, *args, **kwargs):
            if name in ("picamera2", "libcamera") or name.endswith("picamera2_camera"):
                raise AssertionError("Raspberry imports are forbidden for laptop")
            return original(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=guard), \
                patch("face_recognition.infrastructure.camera.opencv_camera.OpenCVCamera") as camera:
            result = create_live_camera("opencv", index=2)
        camera.assert_called_once_with(2, None, None, fps=None, orientation="normal")
        self.assertIs(result, camera.return_value)

    def test_picamera_backend_uses_existing_main_and_lores_adapter_only(self):
        with patch("face_recognition.infrastructure.camera.picamera2_camera.Picamera2Camera") as camera:
            result = create_live_camera("picamera2", inference_size=(480, 270), preview_size=(848, 480))
        camera.assert_called_once_with(480, 270, display_width=848, display_height=480, orientation="normal")
        self.assertIs(result, camera.return_value)

    def test_invalid_backend_rejected_before_camera_creation(self):
        with patch("face_recognition.infrastructure.camera.opencv_camera.OpenCVCamera") as camera:
            with self.assertRaises(ValueError): create_live_camera("invalid")
            camera.assert_not_called()

    def test_native_laptop_capture_does_not_request_pi_dimensions_or_frame_rate(self):
        capture = Mock()
        capture.isOpened.return_value = True
        with patch("face_recognition.infrastructure.camera.opencv_camera.cv2.VideoCapture", return_value=capture):
            camera = OpenCVCamera(0, None, None, fps=None)
        capture.set.assert_not_called()
        camera.close()
        capture.release.assert_called_once()
