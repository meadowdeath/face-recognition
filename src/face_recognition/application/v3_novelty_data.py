"""Read-only populations and temporal observations; no sklearn or file access."""

from dataclasses import dataclass, field
import re

import numpy as np

from face_recognition.application.create_v3_cv_protocol import validate_v3_cv_protocol
from face_recognition.application.temporal_windows import aggregate_windows, build_windows
from face_recognition.domain.models.processed_sample import SampleProvenance
from face_recognition.domain.models.processed_sample import BASELINE_REPRESENTATION, POSE_REPRESENTATION
from face_recognition.domain.models.dataset import CONDITIONS
from face_recognition.domain.models.rejection import RejectionError
from face_recognition.domain.models.temporal import WindowCounts
from face_recognition.domain.models.v3_cv_protocol import V3CVProtocol


def validate_representation_catalog(protocol, catalog):
    """Bind candidate population keys to the unchanged baseline protocol.

    Candidate values/file sizes/manifest hashes differ intentionally. Identity,
    session, condition ordering and declared populations must still agree.
    """
    if catalog.representation == BASELINE_REPRESENTATION:
        if catalog != protocol.catalog:
            raise RejectionError("Baseline catalog does not match protocol")
    elif (catalog.representation != POSE_REPRESENTATION
          or catalog.observed_sessions != protocol.catalog.observed_sessions
          or catalog.declared_source_samples != protocol.catalog.declared_source_samples
          or tuple((r.person_id, r.session_id, r.condition) for r in catalog.condition_files)
          != tuple((r.person_id, r.session_id, r.condition) for r in protocol.catalog.condition_files)):
        raise RejectionError("Candidate population does not match protocol")


@dataclass(frozen=True, eq=False)
class V3Population:
    X: np.ndarray = field(repr=False)
    y: np.ndarray = field(repr=False)
    samples: tuple[SampleProvenance, ...] = field(repr=False)

@dataclass(frozen=True, eq=False)
class V3PayloadData:
    protocol: V3CVProtocol = field(repr=False)
    X: np.ndarray = field(repr=False)
    y: np.ndarray = field(repr=False)
    samples: tuple[SampleProvenance, ...] = field(repr=False)
    payload_sha256: str
    source_stamps: tuple = field(default=(), repr=False)
    representation: str = BASELINE_REPRESENTATION

    def __post_init__(self):
        self.validate()
        self.X.setflags(write=False)
        self.y.setflags(write=False)

    def validate(self):
        validate_v3_cv_protocol(self.protocol)
        if (self.representation not in (BASELINE_REPRESENTATION, POSE_REPRESENTATION)
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
        return V3Population(X, y, tuple(self.samples[i] for i in indices))


@dataclass(frozen=True, eq=False)
class FrameScores:
    predicted: np.ndarray = field(repr=False)
    decisions: np.ndarray = field(repr=False)
    confidence: tuple[tuple[str, np.ndarray], ...] = field(repr=False)


@dataclass(frozen=True, eq=False)
class WindowBatch:
    keys: tuple = field(repr=False)
    scores: np.ndarray = field(repr=False)
    correct: np.ndarray = field(repr=False)
    counts: WindowCounts


class ScoredPopulation:
    """Cache observations, not transformed matrices; subset only complete windows."""
    def __init__(self, population: V3Population, frames: FrameScores):
        self.samples, self.frame_y, self.frame_predicted = population.samples, population.y, frames.predicted
        plan = build_windows(population.samples, 7)
        windows = aggregate_windows(plan, frames.predicted, frames.decisions, frames.confidence)
        self.keys = tuple((w.person_id, w.session_id, w.condition,
                           tuple(population.samples[i].sample_index for i in w.indices)) for w in plan.windows)
        true_y = np.array([population.y[w.indices[0]] for w in plan.windows], dtype=np.int32)
        self.correct = windows.predicted == true_y
        self.confidence = windows.confidence

    def batch(self, method, *, sessions=None, identities=None):
        def accepts(person, session):
            return (sessions is None or session in sessions) and (identities is None or person in identities)
        indices = np.array([i for i, key in enumerate(self.keys) if accepts(key[0], key[1])], dtype=np.intp)
        if not len(indices):
            raise RejectionError("V3 evaluation roles require complete window-7 observations")
        source = sum(accepts(s.person_id, s.session_id) for s in self.samples)
        return WindowBatch(tuple(self.keys[i] for i in indices), dict(self.confidence)[method][indices], self.correct[indices],
                           WindowCounts(source, len(indices) * 7, source - len(indices) * 7, len(indices)))

    def __repr__(self):
        return f"ScoredPopulation(frames={len(self.samples)}, windows={len(self.keys)})"
