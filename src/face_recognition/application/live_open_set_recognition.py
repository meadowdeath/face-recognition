"""Shared single-face observation stream. No camera/vendor code, fitting or calibration."""

from dataclasses import dataclass, field
from math import isfinite
from time import monotonic

import numpy as np

from face_recognition.domain.models.face_landmarks import FaceLandmarks
from face_recognition.domain.models.rejection import RejectionError


@dataclass(frozen=True)
class LiveRecognitionView:
    label: str
    last_score: float | None
    threshold: float
    occupancy: int
    window_size: int
    face: FaceLandmarks | None = field(repr=False)
    frame_candidate: str | None = None
    frame_score: float | None = None
    temporal_candidate: str | None = None
    temporal_decision: str | None = None
    revision: int = 0


@dataclass(frozen=True)
class LiveRecognitionMetrics:
    inference_fps: float
    completed_face_inferences: int
    latency_ms: float | None


class LiveTemporalRecognition:
    def __init__(self, recognizer, absence_timeout=2.0):
        if not isfinite(absence_timeout) or absence_timeout <= 0:
            raise ValueError("Absence timeout must be finite and positive")
        if recognizer.representation != "all_xyz_pose_v2" or recognizer.input_width != 1434 or recognizer.window_size != 7:
            raise RejectionError("Live recognition requires the frozen V3 pose/window policy")
        self.recognizer, self.absence_timeout = recognizer, absence_timeout
        self._pending = []
        self._last = self._current = self._face = self._last_valid_at = None
        self._revision = 0

    def expire(self, now):
        if self._last_valid_at is not None and now - self._last_valid_at >= self.absence_timeout:
            if self._pending or self._last is not None or self._current is not None or self._face is not None:
                self._pending.clear()
                self._last = self._current = self._face = None
                self._revision += 1

    def no_face(self, now):
        # Any observed interruption immediately separates presence segments.
        if self._pending or self._current is not None or self._face is not None:
            self._pending.clear()
            self._current = self._face = None
            self._revision += 1
        self.expire(now)

    def observe(self, observation, face, now):
        self.expire(now)
        self._last_valid_at = now
        self._face, self._current = face, observation
        self._pending.append(observation)
        if len(self._pending) == self.recognizer.window_size:
            self._last = self.recognizer.decide_observations(tuple(self._pending))
            self._pending.clear()
        self._revision += 1

    def view(self):
        decision = None if self._last is None else self._last.decision
        label = ("COLLECTING" if self._face is not None else "NO FACE") if decision is None else (
            decision.identity if decision.identity is not None else "UNKNOWN")
        return LiveRecognitionView(label, None if decision is None else decision.knownness_score,
            self.recognizer.threshold, len(self._pending), self.recognizer.window_size, self._face,
            None if self._current is None else self._current.candidate_identity,
            None if self._current is None else self._current.knownness_score,
            None if self._last is None else self._last.candidate_identity,
            None if decision is None else decision.decision.value, self._revision)


class LiveOpenSetRecognition:
    """Consume each completed landmark timestamp at most once, using latest-result semantics.

    Inference FPS counts successful pose+V3 face observations, not camera reads,
    busy skips, repeated snapshots or empty landmarker callbacks. Latency ends
    after V3/temporal processing and begins at the existing submission timestamp.
    """
    def __init__(self, recognizer, normalizer, extractor, *, absence_timeout=2.0, metrics_interval=1.0, clock=monotonic):
        if not isfinite(metrics_interval) or metrics_interval <= 0:
            raise ValueError("Metrics interval must be finite and positive")
        self.recognizer, self.normalizer, self.extractor = recognizer, normalizer, extractor
        self.state = LiveTemporalRecognition(recognizer, absence_timeout)
        self.clock, self.interval = clock, metrics_interval
        self._seen = -1
        self._count = self._sample_count = 0
        self._sample_time = clock()
        self._fps = 0.
        self._latency = None

    def process(self, snapshot):
        now = self.clock()
        self.state.expire(now)
        result = snapshot.result
        if result is None or result.timestamp_ms <= self._seen:
            return self.state.view()
        self._seen = result.timestamp_ms
        # An old result may belong to an appearance that has already ended.
        if not result.faces or now * 1000 - result.timestamp_ms >= self.state.absence_timeout * 1000:
            self.state.no_face(now)
            return self.state.view()
        face = result.faces[0]  # Landmarker configured num_faces=1; deterministic fallback.
        try:
            pose = self.normalizer.normalize(face)
        except ValueError:
            self.state.no_face(now)
            return self.state.view()
        features = self.extractor.extract(pose)
        with np.errstate(over="ignore", invalid="ignore"):
            values = np.asarray(features.values, dtype=np.float32)  # Same dtype as processed payload loading.
        if values.shape != (1434,) or not np.isfinite(values).all():
            raise RejectionError("Live pose feature contract requires 1434 finite values")
        observation = self.recognizer.infer_observation(values)
        completed = self.clock()
        self.state.observe(observation, face, completed)
        finished = self.clock()
        self._latency = max(0., finished * 1000 - result.timestamp_ms)
        self._count += 1
        return self.state.view()

    def metrics(self):
        now = self.clock()
        elapsed = now - self._sample_time
        if elapsed >= self.interval:
            self._fps = (self._count - self._sample_count) / elapsed
            self._sample_time, self._sample_count = now, self._count
        return LiveRecognitionMetrics(self._fps, self._count, self._latency)
