"""Optional Raspberry Pi camera adapter. Never imports Picamera2 on Windows."""

import sys
from typing import Any

import cv2
import numpy as np


class Picamera2Camera:
    def __init__(self, width: int, height: int) -> None:
        if sys.platform != "linux":
            raise RuntimeError("Picamera2Camera requires Raspberry Pi OS/Linux; use OpenCVCamera on Windows")
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise RuntimeError("Picamera2 is unavailable. Install python3-picamera2 via Raspberry Pi OS packages") from exc
        self._camera: Any = Picamera2()
        self._drm_preview_started = False
        try:
            configuration = self._camera.create_preview_configuration(main={"size": (width, height), "format": "RGB888"})
            self._camera.configure(configuration)
            self._camera.start()
        except BaseException:
            self._camera.close()
            raise

    def read(self) -> np.ndarray | None:
        rgb = self._camera.capture_array()
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb is not None else None

    def start_drm_preview(self, width: int, height: int) -> None:
        from picamera2 import Preview

        # start() already supplies a NULL preview/event loop in Picamera2 0.3.31.
        self._camera.stop_preview()
        self._camera.start_preview(Preview.DRM, width=width, height=height)
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
