"""Guided raw-landmark capture, independent of cameras, JSON and filesystem APIs."""

from enum import Enum

from face_recognition.domain.interfaces.dataset_repository import DatasetRepository
from face_recognition.domain.models.dataset import (
    CONDITIONS, RawLandmarkSample, SessionConfig, validate_raw_landmarks,
)
from face_recognition.domain.models.landmark_result import LandmarkResult

# Separate capture instructions allow small variation, unlike the static validation experiment.
INSTRUCTIONS = (
    "Look directly at the camera, keep your head upright and near the center. During capture, remain approximately in this position but allow small natural movements.",
    "Move your whole head/body toward the left side of the frame while keeping your face pointed toward the camera. Do not deliberately turn your head. Maintain that region with small natural variation.",
    "Move your whole head/body toward the right side of the frame while keeping your face pointed toward the camera. Do not deliberately turn your head. Maintain that region with small natural variation.",
    "Position your whole head higher in the frame while maintaining a frontal face. Maintain that region with small natural variation.",
    "Position your whole head lower in the frame while maintaining a frontal face. Maintain that region with small natural variation.",
    "Move moderately closer to the camera while remaining frontal and upright. Do not move extremely close. Maintain this distance with small natural variation.",
    "Move moderately farther from the camera while remaining frontal and upright. Do not move extremely far away. Maintain this distance with small natural variation.",
    "Tilt your head toward your left shoulder, like an ‘eh?’ gesture, while generally facing the camera. Maintain the tilt with small natural variation.",
    "Tilt your head toward your right shoulder, like an ‘eh?’ gesture, while generally facing the camera. Maintain the tilt with small natural variation.",
    "Keep approximately the same position but rotate your face toward your left, like one side of a ‘no’ gesture. Use a moderate angle, not an extreme profile view. Allow small natural variation.",
    "Keep approximately the same position but rotate your face toward your right, like one side of a ‘no’ gesture. Use a moderate angle, not an extreme profile view. Allow small natural variation.",
    "Raise your chin/look moderately upward while keeping your head approximately in the same location. Allow small natural variation.",
    "Lower your chin/look moderately downward while keeping your head approximately in the same location. Allow small natural variation.",
)


class CapturePhase(Enum):
    WAITING = "waiting"
    WARMUP = "warm-up"
    COLLECTING = "collecting"
    DONE = "done"
    CANCELLED = "cancelled"


class CaptureDataset:
    """One bounded condition batch; only distinct, valid, time-spaced results.

    Completed files survive cancellation. An unfinished batch is discarded and
    recaptured on resume. Detector timestamps are process-local monotonic times;
    spacing is enforced within each condition, not across resumed processes.
    """

    def __init__(self, repository: DatasetRepository, config: SessionConfig) -> None:
        self.repository = repository
        self.config = config
        self.manifest = repository.open_session(config)
        self.stage_index = len(self.manifest.completed_conditions)
        self.phase = CapturePhase.DONE if self.stage_index == len(CONDITIONS) else CapturePhase.WAITING
        self.last_timestamp_ms: int | None = None
        self.last_accepted_timestamp_ms: int | None = None
        self.expected_landmark_count = self.manifest.expected_landmark_count
        self.warmup_count = self.accepted_count = 0
        self.status = ""
        self._samples: list[RawLandmarkSample] = []

    @property
    def condition(self) -> str | None:
        return CONDITIONS[self.stage_index] if self.stage_index < len(CONDITIONS) else None

    def start_condition(self) -> bool:
        if self.phase is not CapturePhase.WAITING:
            return False
        self.warmup_count = self.accepted_count = 0
        self.last_accepted_timestamp_ms = None
        self.status = ""
        self._samples.clear()
        self.phase = CapturePhase.WARMUP if self.config.protocol.warmup_results else CapturePhase.COLLECTING
        return True

    def observe(self, result: LandmarkResult | None) -> bool:
        if result is None or (self.last_timestamp_ms is not None and result.timestamp_ms <= self.last_timestamp_ms):
            return False
        self.last_timestamp_ms = result.timestamp_ms
        if self.phase not in (CapturePhase.WARMUP, CapturePhase.COLLECTING):
            return False
        self.status = ""
        if self.phase is CapturePhase.WARMUP:
            self.warmup_count += 1
            if self.warmup_count == self.config.protocol.warmup_results:
                self.phase = CapturePhase.COLLECTING
            return True
        if len(result.faces) != 1:
            self.status = "No face; waiting." if not result.faces else "Multiple faces; waiting for one participant."
            return True
        face = result.faces[0]
        try:
            validate_raw_landmarks(face, self.expected_landmark_count)
        except ValueError as exc:
            self.status = str(exc)
            return True
        if (self.last_accepted_timestamp_ms is not None
                and result.timestamp_ms - self.last_accepted_timestamp_ms < self.config.protocol.minimum_sample_interval_ms):
            self.status = "Waiting for sample spacing."
            return True
        self.expected_landmark_count = len(face.points)
        self._samples.append(RawLandmarkSample(
            self.config.person_id, self.config.session_id, self.condition, self.accepted_count,
            result.timestamp_ms, self.config.inference_width, self.config.inference_height, face,
        ))
        self.last_accepted_timestamp_ms = result.timestamp_ms
        self.accepted_count += 1
        if self.accepted_count == self.config.protocol.samples_per_condition:
            self.manifest = self.repository.save_condition(self.manifest, self.condition, tuple(self._samples))
            self._samples.clear()
            self.stage_index = len(self.manifest.completed_conditions)
            self.phase = CapturePhase.DONE if self.stage_index == len(CONDITIONS) else CapturePhase.WAITING
        return True

    def cancel(self) -> None:
        self._samples.clear()
        if self.phase is not CapturePhase.DONE:
            self.phase = CapturePhase.CANCELLED
