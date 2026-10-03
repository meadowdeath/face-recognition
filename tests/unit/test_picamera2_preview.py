from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

import numpy as np

from face_recognition.infrastructure.camera import picamera2_camera as module


class Picamera2PreviewTests(unittest.TestCase):
    def make_camera(self, native: Mock, orientation: str = "normal") -> module.Picamera2Camera:
        fake = SimpleNamespace(Picamera2=Mock(return_value=native), Preview=SimpleNamespace(DRM="fake-drm"))
        self.transform = Mock(return_value="fake-transform")
        with patch.object(module.sys, "platform", "linux"), \
             patch.dict("sys.modules", {"picamera2": fake, "libcamera": SimpleNamespace(Transform=self.transform)}):
            camera = module.Picamera2Camera(480, 270, display_width=848, display_height=480,
                                           orientation=orientation)
            camera.start_drm_preview(848, 480)
        return camera

    def test_null_preview_is_replaced_and_drm_resources_are_closed(self) -> None:
        native = Mock()
        camera = self.make_camera(native)
        calls = native.mock_calls
        stop_index = calls.index(call.stop_preview())
        start_index = calls.index(call.start_preview("fake-drm", x=0, y=0, width=848, height=480))
        self.assertLess(stop_index, start_index)
        overlay = np.zeros((480, 848, 4), dtype=np.uint8)
        camera.set_overlay(overlay)
        self.assertIs(native.set_overlay.call_args.args[0], overlay)
        camera.close()
        self.assertEqual(native.mock_calls[-4:],
                         [call.set_overlay(None), call.stop_preview(), call.stop(), call.close()])

    def test_orientation_applies_to_both_configured_streams(self) -> None:
        for orientation, flags in (
            ("normal", {}), ("rotate180", {"hflip": 1, "vflip": 1}),
            ("mirror-horizontal", {"hflip": 1}), ("mirror-vertical", {"vflip": 1}),
        ):
            with self.subTest(orientation=orientation):
                native = Mock()
                camera = self.make_camera(native, orientation)
                self.transform.assert_called_once_with(**flags)
                native.create_preview_configuration.assert_called_once_with(
                    main={"size": (848, 480), "format": "RGB888"},
                    lores={"size": (480, 270), "format": "YUV420", "preserve_ar": False},
                    transform="fake-transform", display="main",
                )
                camera.close()

    def test_read_uses_lores_and_excludes_stride_padding_without_resizing(self) -> None:
        native = Mock()
        native.stream_configuration.return_value = {"size": (480, 270)}
        # I420 with a padded 512-pixel stride, neutral chroma and black luma.
        yuv = np.full((405, 512), 128, dtype=np.uint8)
        yuv[:270] = 16
        native.capture_array.return_value = yuv
        camera = self.make_camera(native)
        with patch.object(module.cv2, "resize") as resize:
            frame = camera.read()
        self.assertEqual(frame.shape, (270, 480, 3))
        self.assertFalse(np.any(frame))
        native.capture_array.assert_called_once_with("lores")
        native.stream_configuration.assert_called_once_with("lores")
        resize.assert_not_called()
        native.capture_array.return_value = None
        self.assertIsNone(camera.read())
        camera.close()

    def test_invalid_configuration_fails_before_creating_camera(self) -> None:
        for kwargs in ({"orientation": "rotate90"}, {"display_width": 320}, {"display_height": 479}):
            with self.subTest(kwargs=kwargs), patch.object(module.sys, "platform", "linux"), \
                 patch.dict("sys.modules", {"picamera2": None, "libcamera": None}):
                with self.assertRaises(ValueError):
                    module.Picamera2Camera(480, 270, **kwargs)

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
