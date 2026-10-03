"""Current frame orchestration. Identity recognition is future work."""

from dataclasses import dataclass

from face_recognition.domain.interfaces.camera import Camera, Frame
from face_recognition.domain.interfaces.landmark_detector import LandmarkDetector
from face_recognition.domain.models.face_landmarks import FaceLandmarks


@dataclass(frozen=True)
class LandmarkFrame:
    frame: Frame
    faces: list[FaceLandmarks]


class LandmarkPreview:
    def __init__(self, camera: Camera, detector: LandmarkDetector) -> None:
        self.camera = camera
        self.detector = detector

    def next_frame(self) -> LandmarkFrame | None:
        frame = self.camera.read()
        if frame is None:
            return None
        return LandmarkFrame(frame, self.detector.detect(frame))

    def close(self) -> None:
        try:
            self.detector.close()
        finally:
            self.camera.close()
