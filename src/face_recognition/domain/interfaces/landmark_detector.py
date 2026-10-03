from typing import Protocol

from face_recognition.domain.interfaces.camera import Frame
from face_recognition.domain.models.landmark_result import DetectorSnapshot


class LandmarkDetector(Protocol):
    def submit(self, frame: Frame) -> None:
        """Submit a recent frame without waiting for inference to finish.

        Implementations may drop frames when busy, rather than queue them.
        """
        ...

    def snapshot(self) -> DetectorSnapshot:
        """Read the latest completed result and cumulative submission counts."""
        ...

    def close(self) -> None: ...
