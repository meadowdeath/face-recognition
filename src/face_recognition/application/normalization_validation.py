"""Pure, in-memory guided normalization validation; no camera or vendor APIs."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from math import fsum, isfinite
from statistics import fmean, median, pstdev

from face_recognition.application.normalization_experiment import (
    LandmarkNormalization, landmark_rmse, raw_comparison_geometry,
)
from face_recognition.domain.models.face_landmarks import FaceLandmarks, Landmark
from face_recognition.domain.models.landmark_result import LandmarkResult


@dataclass(frozen=True)
class ValidationConfig:
    warmup_results: int = 5
    measurement_results: int = 30

    def __post_init__(self) -> None:
        if (type(self.warmup_results) is not int or self.warmup_results < 0
                or type(self.measurement_results) is not int or self.measurement_results < 1):
            raise ValueError("Warm-up must be a nonnegative integer and measurements a positive integer")


@dataclass(frozen=True)
class Condition:
    name: str
    description: str


REFERENCE = Condition(
    "REFERENCE",
    "Center your face in the camera, look directly at the camera, keep your head upright and remain still.",
)
CONDITIONS = (
    Condition("CENTER", "Look directly at the camera, keep your head upright, stay near the center of the frame, and remain still."),
    Condition("TRANSLATION LEFT", "Move your whole head/body toward the left side of the camera frame while keeping your face pointed directly at the camera. Do not turn your head. Hold the position still."),
    Condition("TRANSLATION RIGHT", "Move your whole head/body toward the right side of the camera frame while keeping your face pointed directly at the camera. Do not turn your head. Hold the position still."),
    Condition("TRANSLATION UP", "Position your whole head higher in the camera frame while continuing to look directly at the camera. Do not tilt the head. Hold still."),
    Condition("TRANSLATION DOWN", "Position your whole head lower in the camera frame while continuing to look directly at the camera. Do not tilt the head. Hold still."),
    Condition("NEAR", "Move moderately closer to the camera while keeping your face frontal and upright. Do not move extremely close. Hold the new distance still."),
    Condition("FAR", "Move moderately farther from the camera while keeping your face frontal and upright. Do not move extremely far away. Hold the new distance still."),
    Condition("ROLL LEFT", "Tilt your head toward your left shoulder, like an ‘eh?’ gesture. Keep looking toward the camera and hold the tilt still."),
    Condition("ROLL RIGHT", "Tilt your head toward your right shoulder, like an ‘eh?’ gesture. Keep looking toward the camera and hold the tilt still."),
    Condition("YAW LEFT", "Keep your head approximately in the same place but rotate your face toward your left, like the left part of a ‘no’ gesture. Hold that orientation still."),
    Condition("YAW RIGHT", "Keep your head approximately in the same place but rotate your face toward your right, like the right part of a ‘no’ gesture. Hold that orientation still."),
    Condition("PITCH UP", "Keep your head approximately in the same place and raise your chin/look upward, like part of a ‘yes’ gesture. Hold that orientation still."),
    Condition("PITCH DOWN", "Keep your head approximately in the same place and lower your chin/look downward, like part of a ‘yes’ gesture. Hold that orientation still."),
)
STAGES = (REFERENCE, *CONDITIONS)


def mean_landmark_template(samples: Sequence[FaceLandmarks]) -> FaceLandmarks:
    """Average each corresponding x/y/z coordinate independently."""
    if not samples:
        raise ValueError("A template requires at least one sample")
    count = len(samples[0].points)
    if any(len(face.points) != count for face in samples):
        raise ValueError("Landmark counts must match for template construction")
    return FaceLandmarks(tuple(
        Landmark(*(fsum(getattr(face.points[index], axis) for face in samples) / len(samples)
                   for axis in ("x", "y", "z")))
        for index in range(count)
    ))


@dataclass(frozen=True)
class RMSEStatistics:
    mean: float
    std: float
    median: float
    minimum: float
    maximum: float


def summarize_rmse(values: Sequence[float]) -> RMSEStatistics:
    """Population standard deviation (divide variance by sample count)."""
    if not values or any(not isfinite(value) for value in values):
        raise ValueError("Statistics require a nonempty sequence of finite RMSE values")
    return RMSEStatistics(fmean(values), pstdev(values), median(values), min(values), max(values))


@dataclass(frozen=True)
class ReferenceTemplates:
    raw: FaceLandmarks  # Aspect-corrected width-relative geometry, not identity-normalized.
    normalized: FaceLandmarks


@dataclass(frozen=True)
class ConditionReport:
    condition: Condition
    samples: int
    raw: RMSEStatistics
    normalized: RMSEStatistics


class Phase(Enum):
    WAITING = "waiting"
    WARMUP = "warm-up"
    COLLECTING = "collecting"
    DONE = "done"


class NormalizationValidation:
    """SPACE-gated stages; timestamps are deduplicated across stage boundaries.

    Warm-up discards distinct completions, even when no face is found. Measurement
    counts only finite, normalizable first-face results with corresponding counts.
    Reference samples are used only to build templates, never CENTER statistics.
    """

    def __init__(
        self, normalizer: LandmarkNormalization, image_width: int, image_height: int,
        config: ValidationConfig = ValidationConfig(),
    ) -> None:
        if image_width <= 0 or image_height <= 0:
            raise ValueError("Image dimensions must be positive")
        self.config = config
        self._normalizer = normalizer
        self._width, self._height = image_width, image_height
        self._landmark_count: int | None = None
        self._reference_samples: list[tuple[FaceLandmarks, FaceLandmarks]] = []
        self._raw_errors: list[float] = []
        self._normalized_errors: list[float] = []
        self.stage_index = 0
        self.phase = Phase.WAITING
        self.warmup_count = 0
        self.measurement_count = 0
        self.last_timestamp_ms: int | None = None
        self.reference: ReferenceTemplates | None = None
        self.reports: list[ConditionReport] = []
        self.error: str | None = None

    @property
    def condition(self) -> Condition | None:
        return None if self.phase is Phase.DONE else STAGES[self.stage_index]

    def start_stage(self) -> bool:
        if self.phase is not Phase.WAITING:
            return False
        self.warmup_count = self.measurement_count = 0
        self._reference_samples.clear()
        self._raw_errors.clear()
        self._normalized_errors.clear()
        self.error = None
        self.phase = Phase.WARMUP if self.config.warmup_results else Phase.COLLECTING
        return True

    def observe(self, result: LandmarkResult | None) -> bool:
        """Return True for a fresh active-stage completion, including invalid ones."""
        if result is None or (self.last_timestamp_ms is not None
                              and result.timestamp_ms <= self.last_timestamp_ms):
            return False
        self.last_timestamp_ms = result.timestamp_ms
        if self.phase in (Phase.WAITING, Phase.DONE):
            return False
        self.error = None
        if self.phase is Phase.WARMUP:
            self.warmup_count += 1
            if self.warmup_count == self.config.warmup_results:
                self.phase = Phase.COLLECTING
            return True
        if not result.faces:
            self.error = "No face detected; measurement not counted."
            return True
        try:
            face = result.faces[0]
            if self._landmark_count is not None and len(face.points) != self._landmark_count:
                raise ValueError("Landmark counts must match the reference samples; measurement not counted.")
            raw = raw_comparison_geometry(face, self._width, self._height)
            normalized = self._normalizer.normalize(face)
            if len(normalized.points) != len(raw.points):
                raise ValueError("Normalizer must preserve landmark counts; measurement not counted.")
            if any(not isfinite(value) for cloud in (raw, normalized) for point in cloud.points
                   for value in (point.x, point.y, point.z)):
                raise ValueError("Nonfinite landmark coordinates; measurement not counted.")
            if self.stage_index == 0:
                self._reference_samples.append((raw, normalized))
            else:
                self._raw_errors.append(landmark_rmse(raw, self.reference.raw))
                self._normalized_errors.append(landmark_rmse(normalized, self.reference.normalized))
        except ValueError as exc:
            self.error = str(exc)
            return True
        self._landmark_count = len(raw.points)
        self.measurement_count += 1
        if self.measurement_count == self.config.measurement_results:
            self._finish_stage()
        return True

    def _finish_stage(self) -> None:
        if self.stage_index == 0:
            self.reference = ReferenceTemplates(
                mean_landmark_template([raw for raw, _ in self._reference_samples]),
                mean_landmark_template([normalized for _, normalized in self._reference_samples]),
            )
        else:
            self.reports.append(ConditionReport(
                STAGES[self.stage_index], self.measurement_count,
                summarize_rmse(self._raw_errors), summarize_rmse(self._normalized_errors),
            ))
        self._reference_samples.clear()
        self._raw_errors.clear()
        self._normalized_errors.clear()
        self.stage_index += 1
        self.phase = Phase.WAITING if self.stage_index < len(STAGES) else Phase.DONE
