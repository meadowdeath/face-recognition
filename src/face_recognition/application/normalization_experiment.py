"""In-memory normalization diagnostics, independent of capture and presentation."""

from dataclasses import dataclass
from math import fsum, sqrt
from typing import Protocol

from face_recognition.domain.models.face_landmarks import FaceLandmarks, Landmark
from face_recognition.domain.models.landmark_result import LandmarkResult


class LandmarkNormalization(Protocol):
    def normalize(self, landmarks: FaceLandmarks) -> FaceLandmarks: ...


def raw_comparison_geometry(
    landmarks: FaceLandmarks, image_width: int, image_height: int,
) -> FaceLandmarks:
    """Correct image aspect ratio only; keep translation, scale, and roll."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive")
    ratio = image_height / image_width
    return FaceLandmarks(tuple(Landmark(point.x, point.y * ratio, point.z)
                               for point in landmarks.points))


def landmark_rmse(current: FaceLandmarks, reference: FaceLandmarks) -> float:
    """sqrt(mean squared 3D displacement), with equal ordered landmark counts."""
    if len(current.points) != len(reference.points):
        raise ValueError("Landmark counts must match for corresponding-index RMSE")
    squared_error = fsum(
        (point.x - baseline.x) ** 2 + (point.y - baseline.y) ** 2 + (point.z - baseline.z) ** 2
        for point, baseline in zip(current.points, reference.points)
    )
    return sqrt(squared_error / len(current.points))


@dataclass(frozen=True)
class LandmarkReference:
    raw: FaceLandmarks
    normalized: FaceLandmarks


class NormalizationExperiment:
    """First-face comparison updated once per genuinely new completed result.

    The immutable reference survives no-face results and normalization errors.
    No-face completions clear current observations, preventing stale captures.
    No persistence or identity decision is performed.
    """

    def __init__(
        self, normalizer: LandmarkNormalization, image_width: int, image_height: int,
    ) -> None:
        if image_width <= 0 or image_height <= 0:
            raise ValueError("Image dimensions must be positive")
        self._normalizer = normalizer
        self._width, self._height = image_width, image_height
        self._current_geometry: FaceLandmarks | None = None
        self._reference_geometry: FaceLandmarks | None = None
        self.last_timestamp_ms: int | None = None
        self.observations = 0  # Distinct completed results, including no-face results.
        self.current_raw: FaceLandmarks | None = None
        self.current_normalized: FaceLandmarks | None = None
        self.reference: LandmarkReference | None = None
        self.raw_rmse: float | None = None
        self.normalized_rmse: float | None = None
        self.error: str | None = None

    def observe(self, result: LandmarkResult | None) -> bool:
        if result is None or (self.last_timestamp_ms is not None
                              and result.timestamp_ms <= self.last_timestamp_ms):
            return False
        self.last_timestamp_ms = result.timestamp_ms
        self.observations += 1
        self.current_raw = self.current_normalized = self._current_geometry = None
        self.raw_rmse = self.normalized_rmse = None
        self.error = None
        if not result.faces:
            return True
        self.current_raw = result.faces[0]
        try:
            self.current_normalized = self._normalizer.normalize(self.current_raw)
            self._current_geometry = raw_comparison_geometry(self.current_raw, self._width, self._height)
            if self.reference is not None:
                self.raw_rmse = landmark_rmse(self._current_geometry, self._reference_geometry)
                self.normalized_rmse = landmark_rmse(self.current_normalized, self.reference.normalized)
        except ValueError as exc:
            self.raw_rmse = self.normalized_rmse = None
            self.error = str(exc)
        return True

    def capture_reference(self) -> bool:
        if self.current_raw is None or self.current_normalized is None or self._current_geometry is None:
            return False
        self.reference = LandmarkReference(self.current_raw, self.current_normalized)
        self._reference_geometry = self._current_geometry
        self.raw_rmse = self.normalized_rmse = 0.0
        self.error = None
        return True
