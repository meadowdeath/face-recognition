"""Read only normalized_landmarks_v1; all_xyz_v1 is never required or opened."""

import json
from pathlib import Path

from face_recognition.domain.interfaces.normalization_evaluation import ConditionKey
from face_recognition.domain.models.dataset import CONDITIONS, validate_identifier, validate_raw_landmarks
from face_recognition.domain.models.face_landmarks import FaceLandmarks, Landmark
from face_recognition.domain.models.processed_sample import (
    NORMALIZED_REPRESENTATION, PROCESSED_SCHEMA_VERSION, NormalizedLandmarkSample, SampleProvenance,
)
from face_recognition.infrastructure.persistence.processed_dataset_catalog import (
    ProcessedDatasetCatalog, ProcessedDatasetReadError, reject_json_constant,
)


class NormalizedDatasetReadError(ProcessedDatasetReadError):
    """Safe paths/categories only; original decoder exceptions are suppressed."""


def normalized_from_json(line: str) -> NormalizedLandmarkSample:
    try:
        data = json.loads(line, parse_constant=reject_json_constant)
        if (not isinstance(data, dict) or type(data.get("schema_version")) is not int
                or data["schema_version"] != PROCESSED_SCHEMA_VERSION or data["representation"] != NORMALIZED_REPRESENTATION):
            raise ValueError
        provenance = SampleProvenance(**{name: data[name] for name in SampleProvenance.__dataclass_fields__})
        if data["source_relative_path"] != provenance.source_relative_path:
            raise ValueError
        count, points = data["landmark_count"], data["normalized_landmarks"]
        if (type(count) is not int or count <= 0 or not isinstance(points, list) or len(points) != count
                or any(not isinstance(point, list) or len(point) != 3 for point in points)):
            raise ValueError
        landmarks = FaceLandmarks(tuple(Landmark(*point) for point in points))
        # Reuse the existing space-independent numeric/count validation only.
        validate_raw_landmarks(landmarks, count)
        return NormalizedLandmarkSample(provenance, landmarks)
    except Exception:
        raise NormalizedDatasetReadError("Invalid normalized sample JSON/schema") from None


class NormalizedDatasetReader(ProcessedDatasetCatalog):
    error_type = NormalizedDatasetReadError

    def __init__(self, processed_root: Path) -> None:
        super().__init__(processed_root, NORMALIZED_REPRESENTATION)

    def read_condition(self, key: ConditionKey) -> tuple[NormalizedLandmarkSample, ...]:
        person, session, condition = key
        try:
            validate_identifier(person)
            validate_identifier(session)
            if condition not in CONDITIONS:
                raise ValueError
        except ValueError:
            raise NormalizedDatasetReadError("Invalid normalized condition key") from None
        if self._read_manifest() != self._manifest:
            raise NormalizedDatasetReadError("Normalized manifest changed during evaluation")
        path = self._checked(self.root / person / session / (condition + ".jsonl"))
        try:
            with path.open("r", encoding="utf-8") as stream:
                samples = tuple(normalized_from_json(line) for line in stream)
            if not samples:
                raise ValueError
            for index, sample in enumerate(samples):
                provenance = sample.provenance
                if (provenance.person_id, provenance.session_id, provenance.condition) != key or provenance.sample_index != index:
                    raise ValueError
            return samples
        except Exception:
            raise NormalizedDatasetReadError(f"Invalid normalized condition samples/metadata: {path}") from None
