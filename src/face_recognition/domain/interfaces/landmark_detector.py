from typing import Protocol

from face_recognition.domain.interfaces.camera import Frame
from face_recognition.domain.models.face_landmarks import FaceLandmarks


class LandmarkDetector(Protocol):
    def detect(self, frame: Frame) -> list[FaceLandmarks]: ...

    def close(self) -> None: ...
