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
        configuration = self._camera.create_preview_configuration(main={"size": (width, height), "format": "RGB888"})
        self._camera.configure(configuration)
        self._camera.start()

    def read(self) -> np.ndarray | None:
        rgb = self._camera.capture_array()
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb is not None else None

    def close(self) -> None:
        self._camera.stop()
        self._camera.close()
