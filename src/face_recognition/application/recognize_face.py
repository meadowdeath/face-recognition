"""Current frame orchestration. Identity recognition is future work."""

from dataclasses import dataclass

from face_recognition.application.performance import PerformanceTracker
from face_recognition.domain.interfaces.camera import Camera, Frame
from face_recognition.domain.interfaces.landmark_detector import LandmarkDetector
from face_recognition.domain.models.face_landmarks import FaceLandmarks
from face_recognition.domain.models.landmark_result import DetectorSnapshot


@dataclass(frozen=True)
class LandmarkFrame:
    frame: Frame
    detector_snapshot: DetectorSnapshot

    @property
    def faces(self) -> tuple[FaceLandmarks, ...]:
        result = self.detector_snapshot.result
        return () if result is None else result.faces


class LandmarkPreview:
    def __init__(
        self,
        camera: Camera,
        detector: LandmarkDetector,
        performance: PerformanceTracker | None = None,
    ) -> None:
        self.camera = camera
        self.detector = detector
        self.performance = performance if performance is not None else PerformanceTracker()

    def next_frame(self) -> LandmarkFrame | None:
        frame = self.camera.read()
        if frame is None:
            return None
        self.performance.record_capture()
        self.detector.submit(frame)
        return LandmarkFrame(frame, self.detector.snapshot())

    def close(self) -> None:
        try:
            self.detector.close()
        finally:
            self.camera.close()
