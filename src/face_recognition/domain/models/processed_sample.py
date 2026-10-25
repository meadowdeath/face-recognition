"""Derived representations and processing metadata, distinct from canonical raw samples."""

from dataclasses import dataclass
from math import isfinite

from face_recognition.domain.models.dataset import CONDITIONS, RawLandmarkSample, validate_identifier
from face_recognition.domain.models.face_features import FaceFeatures
from face_recognition.domain.models.face_landmarks import FaceLandmarks

PROCESSED_SCHEMA_VERSION = 1
NORMALIZED_REPRESENTATION = "normalized_landmarks_v1"
BASELINE_REPRESENTATION = "all_xyz_v1"
REPRESENTATIONS = (NORMALIZED_REPRESENTATION, BASELINE_REPRESENTATION)


@dataclass(frozen=True)
class SampleProvenance:
    person_id: str
    session_id: str
    condition: str
    sample_index: int
    detector_timestamp_ms: int
    inference_width: int
    inference_height: int

    def __post_init__(self) -> None:
        validate_identifier(self.person_id)
        validate_identifier(self.session_id)
        if self.condition not in CONDITIONS:
            raise ValueError("Unknown source condition")
        for name in ("sample_index", "detector_timestamp_ms", "inference_width", "inference_height"):
            value = getattr(self, name)
            minimum = 1 if name.startswith("inference_") else 0
            if type(value) is not int or value < minimum:
                raise ValueError("Invalid source index, timestamp or inference dimensions")

    @classmethod
    def from_raw(cls, sample: RawLandmarkSample) -> "SampleProvenance":
        return cls(sample.person_id, sample.session_id, sample.condition, sample.sample_index,
                   sample.detector_timestamp_ms, sample.inference_width, sample.inference_height)

    @property
    def source_relative_path(self) -> str:
        return f"{self.person_id}/{self.session_id}/{self.condition}.jsonl"

    @property
    def condition_key(self) -> tuple[str, str, int]:
        return self.person_id, self.session_id, CONDITIONS.index(self.condition)


@dataclass(frozen=True)
class NormalizedLandmarkSample:
    provenance: SampleProvenance
    landmarks: FaceLandmarks

    def __post_init__(self) -> None:
        if any(not isfinite(value) for point in self.landmarks.points for value in (point.x, point.y, point.z)):
            raise ValueError("Normalized coordinates must be finite")


@dataclass(frozen=True)
class BaselineFeatureSample:
    provenance: SampleProvenance
    features: FaceFeatures
    landmark_count: int

    def __post_init__(self) -> None:
        if type(self.landmark_count) is not int or self.landmark_count < 1:
            raise ValueError("Landmark count must be a positive integer")
        if len(self.features.values) != self.landmark_count * 3:
            raise ValueError("All-XYZ features require three values per landmark")
        if any(not isfinite(value) for value in self.features.values):
            raise ValueError("Feature values must be finite")


@dataclass(frozen=True)
class ProcessingSummary:
    """Counts of written provenance groups; conditions are session/condition files.

    total_samples_processed counts successful representation pairs. File-level
    failures increment error_count; rejected_samples counts a known failing
    record (including a decoder's expected line index), never guessed records.
    """

    participants: int = 0
    sessions: int = 0
    conditions: int = 0
    total_samples_processed: int = 0
    rejected_samples: int = 0
    error_count: int = 0

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 0 for value in self.__dict__.values()):
            raise ValueError("Processing counts must be nonnegative integers")


@dataclass(frozen=True)
class ProcessingIssue:
    category: str
    person_id: str | None = None
    session_id: str | None = None
    condition: str | None = None
    sample_index: int | None = None

    def describe(self) -> str:
        return (f"{self.category}: person_id={self.person_id or 'unknown'}, "
                f"session_id={self.session_id or 'unknown'}, condition={self.condition or 'unknown'}, "
                f"sample_index={self.sample_index if self.sample_index is not None else 'unknown'}")
