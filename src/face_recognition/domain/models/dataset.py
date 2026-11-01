"""Raw landmark dataset concepts; no image processing or filesystem dependencies."""

from dataclasses import dataclass, field
from math import isfinite
import re

from face_recognition.domain.models.face_landmarks import FaceLandmarks

SCHEMA_VERSION = 1
COORDINATE_SPACE = (
    "Raw MediaPipe image-normalized landmarks: x normalized by input width, y normalized by input height; "
    "z approximately in x's scale. Not identity-normalized."
)
CONDITIONS = (
    "center", "translation_left", "translation_right", "translation_up", "translation_down",
    "near", "far", "roll_left", "roll_right", "yaw_left", "yaw_right", "pitch_up", "pitch_down",
)


def validate_identifier(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value):
        raise ValueError("Identifiers must contain 1–64 ASCII letters, numbers, underscores or hyphens")
    if value.upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                          *(f"LPT{i}" for i in range(1, 10))}:
        raise ValueError("Identifiers must not be Windows reserved device names")
    return value


def _integer(value: int, name: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def validate_raw_landmarks(face: FaceLandmarks, expected_count: int | None = None) -> None:
    if expected_count is not None and len(face.points) != expected_count:
        raise ValueError("Landmark count differs from the session's expected count")
    for point in face.points:
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value)
               for value in (point.x, point.y, point.z)):
            raise ValueError("Raw landmark coordinates must be finite numbers")


@dataclass(frozen=True)
class CaptureProtocol:
    warmup_results: int = 5
    samples_per_condition: int = 20
    minimum_sample_interval_ms: int = 250

    def __post_init__(self) -> None:
        _integer(self.warmup_results, "Warm-up results")
        _integer(self.samples_per_condition, "Samples per condition", 1)
        _integer(self.minimum_sample_interval_ms, "Minimum sample interval")


@dataclass(frozen=True)
class SessionConfig:
    person_id: str
    session_id: str
    camera_backend: str
    camera_device_id: str
    orientation: str
    inference_width: int
    inference_height: int
    display_width: int
    display_height: int
    detector_model_sha256: str
    mediapipe_version: str
    detection_confidence: float
    tracking_confidence: float
    requested_camera_fps: int | None = None
    protocol: CaptureProtocol = field(default_factory=CaptureProtocol)

    def __post_init__(self) -> None:
        validate_identifier(self.person_id)
        validate_identifier(self.session_id)
        if self.camera_backend not in ("opencv", "picamera2"):
            raise ValueError("Unsupported camera backend")
        if not isinstance(self.camera_device_id, str) or not self.camera_device_id:
            raise ValueError("Camera device identifier must be a nonempty string")
        if self.orientation not in ("normal", "rotate180", "mirror-horizontal", "mirror-vertical"):
            raise ValueError("Unsupported orientation")
        for name in ("inference_width", "inference_height", "display_width", "display_height"):
            _integer(getattr(self, name), name, 1)
        if self.requested_camera_fps is not None:
            _integer(self.requested_camera_fps, "Requested camera FPS", 1)
        if not isinstance(self.detector_model_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", self.detector_model_sha256):
            raise ValueError("Detector model SHA256 must contain 64 lowercase hexadecimal characters")
        if not isinstance(self.mediapipe_version, str) or not self.mediapipe_version:
            raise ValueError("MediaPipe version must be a nonempty string")
        for value in (self.detection_confidence, self.tracking_confidence):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Detector confidences must be finite numbers between 0 and 1")


@dataclass(frozen=True)
class SessionManifest:
    config: SessionConfig
    created_at: str
    expected_landmark_count: int | None = None
    completed_conditions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.expected_landmark_count is not None:
            _integer(self.expected_landmark_count, "Expected landmark count", 1)
        if self.completed_conditions != CONDITIONS[:len(self.completed_conditions)]:
            raise ValueError("Completed conditions must follow the capture protocol order")
        if self.completed_conditions and self.expected_landmark_count is None:
            raise ValueError("Completed conditions require an established landmark count")


@dataclass(frozen=True)
class RawLandmarkSample:
    person_id: str
    session_id: str
    condition: str
    sample_index: int
    detector_timestamp_ms: int
    inference_width: int
    inference_height: int
    landmarks: FaceLandmarks

    def __post_init__(self) -> None:
        validate_identifier(self.person_id)
        validate_identifier(self.session_id)
        if self.condition not in CONDITIONS:
            raise ValueError("Unknown capture condition")
        _integer(self.sample_index, "Sample index")
        _integer(self.detector_timestamp_ms, "Detector timestamp")
        _integer(self.inference_width, "Inference width", 1)
        _integer(self.inference_height, "Inference height", 1)
        validate_raw_landmarks(self.landmarks)
