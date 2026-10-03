"""MediaPipe Tasks video-mode detector for BGR camera frames."""

from pathlib import Path
from time import monotonic_ns

import cv2
import mediapipe as mp
import numpy as np

from face_recognition.domain.models.face_landmarks import FaceLandmarks, Landmark


class MediaPipeFaceLandmarker:
    def __init__(
        self,
        model_path: Path,
        max_faces: int,
        detection_confidence: float,
        tracking_confidence: float,
    ) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Face Landmarker model missing: {model_path}. "
                "Run python tools/download_face_landmarker.py first."
            )
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=max_faces,
            min_face_detection_confidence=detection_confidence,
            min_face_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def detect(self, frame: np.ndarray) -> list[FaceLandmarks]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        timestamp_ms = max(monotonic_ns() // 1_000_000, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        result = self._detector.detect_for_video(image, timestamp_ms)
        return [
            FaceLandmarks(tuple(Landmark(point.x, point.y, point.z) for point in face))
            for face in result.face_landmarks
        ]

    def close(self) -> None:
        self._detector.close()
