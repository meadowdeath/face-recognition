"""Hardware-free observations and a loaded-recognizer adapter; never fit a model."""

from types import SimpleNamespace

import numpy as np

from face_recognition.application.v3_deployable_recognition import V3DeployableRecognizer
from face_recognition.application.v3_novelty_data import FrameScores
from face_recognition.domain.models.landmark_result import DetectorSnapshot, LandmarkResult
from face_recognition.domain.models.v3_deployable_model import DeployableV3Model, V3FrameObservation
from tests.pose_representation_fakes import encoded_face, face_geometry

CLASSES = ("known_a", "known_b", "known_c")


class Clock:
    def __init__(self): self.now = 100.
    def __call__(self): return self.now
    def advance(self, value): self.now += value


def observation(index=0, score=-12., decisions=None):
    return V3FrameObservation(index, CLASSES[index], score,
        tuple(decisions if decisions is not None else (2. if i == index else 0. for i in range(3))))


class InferenceOnlyRunner:
    def __init__(self, observations=(), clock=None, delay=0.):
        self.observations = list(observations)
        self.clock, self.delay = clock, delay
        self.fit_count = self.infer_calls = 0
        self.inputs = []

    def fit(self, *args, **kwargs): raise AssertionError("Live code must never fit")
    def model_identity(self, model): return (id(model),)
    def fingerprints(self, model): return model.version

    def infer(self, model, population):
        self.infer_calls += 1
        self.inputs.append(population.X.copy())
        rows = [self.observations.pop(0) if self.observations else observation() for _ in population.X]
        if self.clock is not None: self.clock.advance(self.delay)
        return FrameScores(np.array([r.candidate_index for r in rows], dtype=np.int32),
            np.array([r.decision_scores for r in rows], dtype=float),
            (("mahalanobis_distance", np.array([r.knownness_score for r in rows], dtype=float)),))


def loaded_recognizer(observations=(), threshold=-18., clock=None, delay=0.):
    runner = InferenceOnlyRunner(observations, clock, delay)
    config = SimpleNamespace(pipeline=SimpleNamespace(representation="all_xyz_pose_v2", input_features=1434,
        novelty_method="mahalanobis_distance"), temporal=SimpleNamespace(window_size=7))
    model = DeployableV3Model(SimpleNamespace(identity_count=3, version=1), config, CLASSES, threshold)
    return V3DeployableRecognizer(model, runner, 1), runner


def face(): return encoded_face(face_geometry())


def snapshot(clock, faces=None, *, timestamp_ms=None, completed=1):
    return DetectorSnapshot(LandmarkResult((face(),) if faces is None else faces,
        int(clock.now * 1000) if timestamp_ms is None else timestamp_ms, 0.), completed_frames=completed)
