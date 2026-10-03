import sys

import cv2
import numpy as np

from face_recognition.config.settings import CAMERA_ORIENTATIONS


class OpenCVCamera:
    def __init__(self, index: int, width: int, height: int, fps: int = 30, *, orientation: str = "normal") -> None:
        if orientation not in CAMERA_ORIENTATIONS:
            raise ValueError(f"Unsupported camera orientation: {orientation}")
        self._flip_code = {"normal": None, "rotate180": -1, "mirror-horizontal": 1, "mirror-vertical": 0}[orientation]
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        self._capture = cv2.VideoCapture(index, backend)
        if not self._capture.isOpened():
            self._capture.release()
            raise RuntimeError(f"Could not open camera index {index}")
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # This is a request; the device/backend may choose a different rate.
        self._capture.set(cv2.CAP_PROP_FPS, fps)

    def read(self) -> np.ndarray | None:
        ok, frame = self._capture.read()
        if not ok:
            return None
        return frame if self._flip_code is None else cv2.flip(frame, self._flip_code)

    def close(self) -> None:
        self._capture.release()
