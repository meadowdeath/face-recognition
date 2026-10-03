"""Optional Raspberry Pi camera adapter. Never imports Picamera2 on Windows."""

import sys
from typing import Any

import cv2
import numpy as np

from face_recognition.config.settings import CAMERA_ORIENTATIONS


class Picamera2Camera:
    def __init__(
        self, width: int, height: int, *, display_width: int | None = None,
        display_height: int | None = None, orientation: str = "normal",
    ) -> None:
        main_width = width if display_width is None else display_width
        main_height = height if display_height is None else display_height
        if orientation not in CAMERA_ORIENTATIONS:
            raise ValueError(f"Unsupported camera orientation: {orientation}")
        if any(value <= 0 or value % 2 for value in (width, height, main_width, main_height)):
            raise ValueError("Picamera2 stream dimensions must be positive even integers")
        if width > main_width or height > main_height:
            raise ValueError("Picamera2 inference dimensions must not exceed main/display dimensions")
        if sys.platform != "linux":
            raise RuntimeError("Picamera2Camera requires Raspberry Pi OS/Linux; use OpenCVCamera on Windows")
        try:
            from picamera2 import Picamera2
            from libcamera import Transform
        except ImportError as exc:
            raise RuntimeError("Picamera2/libcamera is unavailable. Install python3-picamera2 and python3-libcamera via Raspberry Pi OS packages") from exc
        transform_options = {
            "normal": {},
            "rotate180": {"hflip": 1, "vflip": 1},
            "mirror-horizontal": {"hflip": 1},
            "mirror-vertical": {"vflip": 1},
        }
        transform = Transform(**transform_options[orientation])
        self._camera: Any = Picamera2()
        self._drm_preview_started = False
        try:
            configuration = self._camera.create_preview_configuration(
                main={"size": (main_width, main_height), "format": "RGB888"},
                # Pi 3's VC4 pipeline requires YUV lores. Share the main crop
                # so normalized inference coordinates also map to the preview.
                lores={"size": (width, height), "format": "YUV420", "preserve_ar": False},
                transform=transform,
                display="main",
            )
            self._camera.configure(configuration)
            self._camera.start()
        except BaseException:
            self._camera.close()
            raise

    def read(self) -> np.ndarray | None:
        yuv = self._camera.capture_array("lores")
        if yuv is None:
            return None
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        width, height = self._camera.stream_configuration("lores")["size"]
        # capture_array includes YUV stride padding; exclude it after conversion.
        return bgr[:height, :width]

    def start_drm_preview(self, width: int, height: int) -> None:
        from picamera2 import Preview

        # start() already supplies a NULL preview/event loop in Picamera2 0.3.31.
        self._camera.stop_preview()
        self._camera.start_preview(Preview.DRM, x=0, y=0, width=width, height=height)
        self._drm_preview_started = True

    def set_overlay(self, overlay: np.ndarray | None) -> None:
        self._camera.set_overlay(overlay)

    def stop_preview(self) -> None:
        if self._drm_preview_started:
            self._camera.stop_preview()
            self._drm_preview_started = False

    def close(self) -> None:
        try:
            if self._drm_preview_started:
                try:
                    self.set_overlay(None)
                finally:
                    self.stop_preview()
        finally:
            try:
                self._camera.stop()
            finally:
                self._camera.close()
