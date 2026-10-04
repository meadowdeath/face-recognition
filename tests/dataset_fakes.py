from dataclasses import replace
from collections.abc import Sequence

from face_recognition.domain.models.dataset import RawLandmarkSample, SessionConfig, SessionManifest
from face_recognition.domain.models.face_landmarks import FaceLandmarks, Landmark


def session_config(**overrides) -> SessionConfig:
    config = SessionConfig("p001", "session_01", "opencv", "0", "normal", 480, 270, 848, 480,
                           "a" * 64, "0.10.14", 0.5, 0.5, 30)
    return replace(config, **overrides)


def raw_face(count: int = 2) -> FaceLandmarks:
    return FaceLandmarks(tuple(Landmark(0.1234567891234567 + index, -0.25 - index, 0.03125 * index)
                               for index in range(count)))


def condition_samples(config: SessionConfig, condition: str = "center", landmark_count: int = 2) -> tuple[RawLandmarkSample, ...]:
    return tuple(RawLandmarkSample(config.person_id, config.session_id, condition, index,
                                  1000 + index * max(1, config.protocol.minimum_sample_interval_ms),
                                  config.inference_width, config.inference_height, raw_face(landmark_count))
                 for index in range(config.protocol.samples_per_condition))


class FakeDatasetRepository:
    def __init__(self, config: SessionConfig, completed: tuple[str, ...] = ()) -> None:
        self.manifest = SessionManifest(config, "2026-10-03T00:00:00+00:00", 2 if completed else None, completed)
        self.writes: list[tuple[str, tuple[RawLandmarkSample, ...]]] = []

    def open_session(self, config: SessionConfig) -> SessionManifest:
        if self.manifest.config != config:
            raise ValueError("Incompatible fake session")
        return self.manifest

    def save_condition(self, manifest: SessionManifest, condition: str,
                       samples: Sequence[RawLandmarkSample]) -> SessionManifest:
        self.writes.append((condition, tuple(samples)))
        self.manifest = replace(manifest, expected_landmark_count=len(samples[0].landmarks.points),
                                completed_conditions=(*manifest.completed_conditions, condition))
        return self.manifest
