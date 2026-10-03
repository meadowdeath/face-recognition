"""Vendor-independent snapshots of asynchronous landmark detection."""

from dataclasses import dataclass

from face_recognition.domain.models.face_landmarks import FaceLandmarks


@dataclass(frozen=True)
class LandmarkResult:
    """Faces, submission timestamp, and submission-to-callback latency in ms."""

    faces: tuple[FaceLandmarks, ...]
    timestamp_ms: int
    latency_ms: float


@dataclass(frozen=True)
class DetectorSnapshot:
    """Latest completion plus successful submissions, completions and busy skips.

    None means no completion yet; an empty faces tuple means no face found.
    """

    result: LandmarkResult | None = None
    submitted_frames: int = 0
    completed_frames: int = 0
    skipped_busy_frames: int = 0
    # Age measured since the result's submission timestamp, not its callback.
    result_age_ms: float | None = None
    # Diagnostic intervals for the latest completed result, in milliseconds.
    preprocessing_ms: float | None = None
    dispatch_call_ms: float | None = None
    async_result_ms: float | None = None
    total_callback_latency_ms: float | None = None
