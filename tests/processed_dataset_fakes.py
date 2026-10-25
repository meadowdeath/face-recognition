"""Synthetic-only fixtures with enough landmarks for the production normalizer."""

import json
from pathlib import Path

from face_recognition.domain.models.dataset import CONDITIONS, CaptureProtocol, RawLandmarkSample, SessionManifest
from face_recognition.domain.models.face_landmarks import FaceLandmarks, Landmark
from face_recognition.infrastructure.persistence.dataset_repository import manifest_to_dict, sample_to_json
from tests.dataset_fakes import session_config


def synthetic_sample(person: str = "subject_test", session: str = "visit_test", condition: str = "center",
                     index: int = 0, width: int = 480, height: int = 270, count: int = 400) -> RawLandmarkSample:
    # Isotropic geometry is encoded using MediaPipe's different X/Z and Y scales.
    geometry = [Landmark(0.3 + (i % 13) * 0.01, 0.25 + (i % 17) * 0.01, (i % 7) * 0.01)
                for i in range(count)]
    geometry[33] = geometry[133] = Landmark(0.4, 0.2, 0.01)
    geometry[362] = geometry[263] = Landmark(0.6, 0.3, 0.03)
    face = FaceLandmarks(tuple(Landmark(p.x, p.y * width / height, p.z) for p in geometry))
    return RawLandmarkSample(person, session, condition, index, 1000 + 250 * index, width, height, face)


def write_synthetic_session(root: Path, person: str = "subject_test", session: str = "visit_test",
                            completed: tuple[str, ...] = CONDITIONS[:2], width: int = 480,
                            height: int = 270, count: int = 400) -> Path:
    config = session_config(person_id=person, session_id=session, inference_width=width, inference_height=height,
                            protocol=CaptureProtocol(5, 2, 250))
    manifest = SessionManifest(config, "2026-10-24T12:00:00+00:00", count, completed)
    directory = root / person / session
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(json.dumps(manifest_to_dict(manifest)), encoding="utf-8")
    for condition in reversed(completed):
        lines = [sample_to_json(synthetic_sample(person, session, condition, index, width, height, count))
                 for index in range(2)]
        (directory / (condition + ".jsonl")).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return directory


class MemoryProcessedRepository:
    def __init__(self) -> None:
        self.started = False
        self.samples = []
        self.summary = None
        self.failure = None

    def begin(self, source_root: Path) -> None:
        self.started = True

    def write_sample(self, normalized, baseline) -> None:
        self.samples.append((normalized, baseline))

    def complete(self, summary) -> None:
        self.summary = summary

    def fail(self, summary, issue) -> None:
        self.failure = (summary, issue)
