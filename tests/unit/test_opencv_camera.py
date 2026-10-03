import importlib
import unittest
from unittest.mock import Mock, patch

import numpy as np

from face_recognition.infrastructure.camera import opencv_camera as module
from face_recognition.infrastructure.camera import picamera2_camera as pi_adapter


class OpenCVCameraTests(unittest.TestCase):
    def test_orientation_is_applied_to_captured_frames(self) -> None:
        frame = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        for orientation, expected in (
            ("normal", frame), ("rotate180", frame[::-1, ::-1]),
            ("mirror-horizontal", frame[:, ::-1]), ("mirror-vertical", frame[::-1]),
        ):
            with self.subTest(orientation=orientation):
                native = Mock()
                native.read.return_value = (True, frame)
                with patch.object(module.cv2, "VideoCapture", return_value=native):
                    camera = module.OpenCVCamera(0, 848, 480, orientation=orientation)
                np.testing.assert_array_equal(camera.read(), expected)
                native.read.return_value = (False, None)
                self.assertIsNone(camera.read())
                camera.close()
                native.release.assert_called_once()

    def test_camera_modules_import_and_opencv_runs_without_pi_packages(self) -> None:
        native = Mock()
        native.read.return_value = (True, np.zeros((480, 848, 3), dtype=np.uint8))
        with patch.dict("sys.modules", {"picamera2": None, "libcamera": None}):
            importlib.reload(pi_adapter)
            importlib.reload(module)
            with patch.object(module.cv2, "VideoCapture", return_value=native):
                camera = module.OpenCVCamera(0, 848, 480)
                self.assertEqual(camera.read().shape, (480, 848, 3))
                camera.close()

