"""Small, purely synthetic all-XYZ fixtures; no real data or learned models."""

from dataclasses import asdict
import json
from pathlib import Path

from face_recognition.domain.models.dataset import CONDITIONS
from face_recognition.domain.models.processed_sample import (
    BASELINE_REPRESENTATION, PROCESSED_SCHEMA_VERSION, BaselineFeatureSample, ProcessingSummary, SampleProvenance,
)
from face_recognition.infrastructure.features.landmark_feature_extractor import LandmarkFeatureExtractor
from face_recognition.infrastructure.persistence.processed_dataset_repository import baseline_to_json
from tests.dataset_fakes import raw_face


def write_feature_dataset(processed: Path, people: dict[str, tuple[str, ...]],
                          conditions: tuple[str, ...] = CONDITIONS[:2], samples: int = 2, landmarks: int = 3) -> Path:
    root = processed / BASELINE_REPRESENTATION
    root.mkdir(parents=True)
    features = LandmarkFeatureExtractor().extract(raw_face(landmarks))
    for person in reversed(tuple(people)):
        for session in reversed(people[person]):
            directory = root / person / session
            directory.mkdir(parents=True)
            for condition in reversed(conditions):
                lines = []
                for index in range(samples):
                    provenance = SampleProvenance(person, session, condition, index, 1000 + 250 * index, 480, 270)
                    lines.append(baseline_to_json(BaselineFeatureSample(provenance, features, landmarks)))
                (directory / (condition + ".jsonl")).write_text("\n".join(lines) + "\n", encoding="utf-8")
    session_count = sum(len(sessions) for sessions in people.values())
    summary = ProcessingSummary(len(people), session_count, session_count * len(conditions),
                                session_count * len(conditions) * samples)
    header = {"schema_version": PROCESSED_SCHEMA_VERSION, "representation": BASELINE_REPRESENTATION,
              "completion_state": "completed", "condition_files": summary.conditions, **asdict(summary)}
    (root / "processing_manifest.json").write_text(json.dumps(header), encoding="utf-8")
    return root


class MemoryFeatureMetadataSource:
    representation = BASELINE_REPRESENTATION

    def __init__(self, records) -> None:
        self.records = tuple(records)
        self.expected_sample_count = len(self.records)

    def iter_metadata(self):
        yield from self.records
