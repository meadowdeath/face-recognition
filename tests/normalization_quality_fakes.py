"""Only synthetic geometry, serialized with the real raw/processed implementation."""

from dataclasses import replace
import json
from math import cos, sin
from pathlib import Path

from face_recognition.application.normalization_experiment import raw_comparison_geometry
from face_recognition.application.process_dataset import DatasetProcessor
from face_recognition.domain.models.dataset import CONDITIONS, CaptureProtocol, SessionManifest
from face_recognition.domain.models.face_landmarks import FaceLandmarks, Landmark
from face_recognition.infrastructure.features.landmark_feature_extractor import LandmarkFeatureExtractor
from face_recognition.infrastructure.features.landmark_normalizer import LandmarkNormalizer
from face_recognition.infrastructure.persistence.dataset_repository import manifest_to_dict, sample_to_json
from face_recognition.infrastructure.persistence.processed_dataset_repository import FilesystemProcessedDatasetRepository
from face_recognition.infrastructure.persistence.raw_dataset_reader import RawDatasetReader
from tests.dataset_fakes import session_config
from tests.processed_dataset_fakes import synthetic_sample


def write_quality_session(
    root: Path, person: str = "synthetic_subject", session: str = "synthetic_visit",
    transforms: dict | None = None, center_offsets: tuple[float, ...] = (0.0, 0.0),
    width: int = 480, height: int = 270, count: int = 400, identity_shape: float = 0.0,
    conditions: tuple[str, ...] = CONDITIONS,
) -> Path:
    transforms = transforms or {}
    config = session_config(person_id=person, session_id=session, inference_width=width, inference_height=height,
                            protocol=CaptureProtocol(5, len(center_offsets), 250))
    directory = root / person / session
    directory.mkdir(parents=True)
    manifest = SessionManifest(config, "2026-10-24T12:00:00+00:00", count if conditions else None, conditions)
    (directory / "manifest.json").write_text(json.dumps(manifest_to_dict(manifest)), encoding="utf-8")
    for condition in conditions:
        lines = []
        for index in range(len(center_offsets)):
            sample = synthetic_sample(person, session, condition, index, width, height, count)
            geometry = raw_comparison_geometry(sample.landmarks, width, height)
            parameters = transforms.get(condition, {})
            if isinstance(parameters, tuple):
                parameters = parameters[index]
            scale, roll = parameters.get("scale", 1.0), parameters.get("roll", 0.0)
            yaw, pitch = parameters.get("yaw", 0.0), parameters.get("pitch", 0.0)
            tx, ty, tz = parameters.get("translation", (0.0, 0.0, 0.0))
            if condition == "center":
                tx += center_offsets[index]
            points = []
            for point_index, point in enumerate(geometry.points):
                x, y, z = point.x, point.y, point.z + (identity_shape if point_index == 0 else 0.0)
                x, z = cos(yaw) * x + sin(yaw) * z, -sin(yaw) * x + cos(yaw) * z
                y, z = cos(pitch) * y - sin(pitch) * z, sin(pitch) * y + cos(pitch) * z
                x, y = cos(roll) * x - sin(roll) * y, sin(roll) * x + cos(roll) * y
                points.append(Landmark(scale * x + tx, (scale * y + ty) * width / height, scale * z + tz))
            lines.append(sample_to_json(replace(sample, landmarks=FaceLandmarks(tuple(points)))))
        (directory / (condition + ".jsonl")).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return directory


def process_quality_fixture(raw: Path, processed: Path) -> None:
    processor = DatasetProcessor(LandmarkNormalizer, LandmarkFeatureExtractor())
    processor.process(RawDatasetReader(raw), FilesystemProcessedDatasetRepository(processed, raw))


def snapshot(directory: Path) -> dict:
    return {path.relative_to(directory).as_posix():
            (path.read_bytes() if path.is_file() else None, path.stat().st_mtime_ns)
            for path in (directory, *directory.rglob("*"))}
