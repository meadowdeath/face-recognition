from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

import numpy as np

from face_recognition.infrastructure.camera import picamera2_camera as module


class Picamera2PreviewTests(unittest.TestCase):
    def make_camera(self, native: Mock) -> module.Picamera2Camera:
        fake = SimpleNamespace(Picamera2=Mock(return_value=native), Preview=SimpleNamespace(DRM="fake-drm"))
        with patch.object(module.sys, "platform", "linux"), \
             patch.dict("sys.modules", {"picamera2": fake}):
            camera = module.Picamera2Camera(640, 480)
            camera.start_drm_preview(640, 480)
        return camera

    def test_null_preview_is_replaced_and_drm_resources_are_closed(self) -> None:
        native = Mock()
        camera = self.make_camera(native)
        calls = native.mock_calls
        stop_index = calls.index(call.stop_preview())
        start_index = calls.index(call.start_preview("fake-drm", width=640, height=480))
        self.assertLess(stop_index, start_index)
        overlay = np.zeros((480, 640, 4), dtype=np.uint8)
        camera.set_overlay(overlay)
        self.assertIs(native.set_overlay.call_args.args[0], overlay)
        camera.close()
        self.assertEqual(native.mock_calls[-4:],
                         [call.set_overlay(None), call.stop_preview(), call.stop(), call.close()])

    def test_stop_preview_is_not_repeated_after_display_cleanup(self) -> None:
        native = Mock()
        camera = self.make_camera(native)
        camera.set_overlay(None)
        camera.stop_preview()
        native.reset_mock()
        camera.close()
        native.stop_preview.assert_not_called()
        native.set_overlay.assert_not_called()
        native.stop.assert_called_once()
        native.close.assert_called_once()

    def test_overlay_clear_error_still_attempts_every_shutdown_step(self) -> None:
        native = Mock()
        camera = self.make_camera(native)
        native.set_overlay.side_effect = RuntimeError("clear failed")
        with self.assertRaisesRegex(RuntimeError, "clear failed"):
            camera.close()
        self.assertEqual(native.mock_calls[-4:],
                         [call.set_overlay(None), call.stop_preview(), call.stop(), call.close()])
