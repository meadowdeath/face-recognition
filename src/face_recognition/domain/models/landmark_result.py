"""Vendor-independent snapshots of asynchronous landmark detection."""

from dataclasses import dataclass

from face_recognition.domain.models.face_landmarks import FaceLandmarks


@dataclass(frozen=True)
class LandmarkResult:
    """Completed faces and a monotonic submission timestamp in milliseconds."""

    faces: tuple[FaceLandmarks, ...]
    timestamp_ms: int
    latency_ms: float


@dataclass(frozen=True)
class DetectorSnapshot:
    """None means no completion yet; an empty faces tuple means no face found."""

    result: LandmarkResult | None = None
    submitted_frames: int = 0
    completed_frames: int = 0
