"""Read-only populations and temporal observations; no sklearn or file access."""

from dataclasses import dataclass, field
import re

import numpy as np

from face_recognition.application.temporal_windows import aggregate_windows, build_windows
from face_recognition.application.recognition_config import metadata_sha256
from face_recognition.domain.models.processed_sample import SampleProvenance
from face_recognition.domain.models.processed_sample import POSE_REPRESENTATION
from face_recognition.domain.models.dataset import CONDITIONS
from face_recognition.domain.models.dynamic_population import DynamicTrainingProtocol
from face_recognition.domain.models.rejection import RejectionError
from face_recognition.domain.models.temporal import WindowCounts
from face_recognition.domain.models.training_dataset import TrainingProtocol


@dataclass(frozen=True, eq=False)
class Population:
    X: np.ndarray = field(repr=False)
    y: np.ndarray = field(repr=False)
    samples: tuple[SampleProvenance, ...] = field(repr=False)

@dataclass(frozen=True, eq=False)
class TrainingData:
    protocol: TrainingProtocol | DynamicTrainingProtocol = field(repr=False)
    X: np.ndarray = field(repr=False)
    y: np.ndarray = field(repr=False)
    samples: tuple[SampleProvenance, ...] = field(repr=False)
    payload_sha256: str
    source_stamps: tuple = field(default=(), repr=False)
    representation: str = POSE_REPRESENTATION

    def __post_init__(self):
        self.validate()
        self.X.setflags(write=False)
        self.y.setflags(write=False)

    def validate(self):
        if (metadata_sha256(self.protocol.catalog) != self.protocol.catalog_sha256
                or self.representation != POSE_REPRESENTATION
                or self.X.dtype != np.float32 or self.X.shape != (len(self.samples), 1434)
                or not np.isfinite(self.X).all() or self.y.dtype != np.int32 or self.y.shape != (len(self.samples),)
                or len(self.samples) != self.protocol.catalog.declared_source_samples
                or not isinstance(self.payload_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", self.payload_sha256)):
            raise RejectionError("Invalid V3 payload matrices or sample counts")
        labels = {person: i for i, person in enumerate(self.protocol.specification.known_ids)}
        allowed = {(r.person_id, r.session_id, r.condition) for r in self.protocol.catalog.condition_files}
        seen, groups, next_indices, timestamps, dimensions = set(), set(), {}, {}, {}
        previous_key = None
        for y, sample in zip(self.y, self.samples):
            group = sample.person_id, sample.session_id, sample.condition
            key = sample.person_id, sample.session_id, CONDITIONS.index(sample.condition), sample.sample_index
            if (group not in allowed or key in seen or (previous_key is not None and key <= previous_key)
                    or sample.sample_index != next_indices.get(group, 0) or y != labels.get(sample.person_id, -1)):
                raise RejectionError("V3 sample metadata/labels/order do not match protocol")
            if (sample.detector_timestamp_ms <= timestamps.get(group, -1)
                    or dimensions.setdefault(group[:2], (sample.inference_width, sample.inference_height))
                    != (sample.inference_width, sample.inference_height)):
                raise RejectionError("Inconsistent V3 source timestamps or inference dimensions")
            timestamps[group] = sample.detector_timestamp_ms
            previous_key = key
            next_indices[group] = sample.sample_index + 1
            seen.add(key)
            groups.add(group)
        if groups != allowed:
            raise RejectionError("Missing V3 condition payload population")

    def known(self, sessions):
        if not sessions or len(set(sessions)) != len(sessions) or not set(sessions) <= set(self.protocol.specification.known_sessions):
            raise RejectionError("Unauthorized V3 known sessions")
        return self._select(lambda s, y: y >= 0 and s.session_id in sessions)

    def unknown(self, identities):
        if not identities or len(set(identities)) != len(identities) or not set(identities) <= set(self.protocol.specification.unknown_ids):
            raise RejectionError("Unauthorized V3 unknown identities")
        return self._select(lambda s, y: y == -1 and s.person_id in identities)

    def _select(self, predicate):
        indices = np.fromiter((i for i, (s, y) in enumerate(zip(self.samples, self.y)) if predicate(s, y)), dtype=np.intp)
        if not len(indices):
            raise RejectionError("Empty V3 role population")
        X, y = self.X[indices], self.y[indices]
        X.setflags(write=False)
        y.setflags(write=False)
        return Population(X, y, tuple(self.samples[i] for i in indices))


@dataclass(frozen=True, eq=False)
class FrameScores:
    predicted: np.ndarray = field(repr=False)
    decisions: np.ndarray = field(repr=False)
    confidence: tuple[tuple[str, np.ndarray], ...] = field(repr=False)


@dataclass(frozen=True, eq=False)
class WindowBatch:
    scores: np.ndarray = field(repr=False)
    correct: np.ndarray = field(repr=False)
    counts: WindowCounts


def calibration_windows(population: Population, frames: FrameScores, method):
    """Use the frozen seven-observation vote and median once per complete group."""
    plan = build_windows(population.samples, 7)
    windows = aggregate_windows(plan, frames.predicted, frames.decisions, frames.confidence)
    if not plan.windows:
        raise RejectionError("Calibration requires complete window-7 observations")
    true_y = np.array([population.y[w.indices[0]] for w in plan.windows], dtype=np.int32)
    return WindowBatch(dict(windows.confidence)[method], windows.predicted == true_y, plan.counts)
