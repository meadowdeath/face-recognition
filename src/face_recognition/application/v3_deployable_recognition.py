"""Window-level inference only. UNKNOWN never emits an arbitrary registered ID."""

from math import isfinite

import numpy as np

from face_recognition.application.temporal_windows import vote_identity
from face_recognition.application.v3_novelty_data import V3Population
from face_recognition.domain.models.rejection import RejectionError
from face_recognition.domain.models.v3_deployable_model import IdentityDecision, V3RecognitionDecision


class V3DeployableRecognizer:
    """One coherent seven-frame feature window; callers own capture boundaries.

    The threshold is calibrated for windows, so there is deliberately no
    operational single-frame threshold API or live camera/buffer implementation.
    """
    def __init__(self, model, runner, expected_fingerprints):
        self._model, self._runner = model, runner
        self._expected = expected_fingerprints
        self._generation = runner.fit_count
        self._identity = runner.model_identity(model.fitted)
        if len(model.class_order) != model.fitted.identity_count or not isfinite(model.threshold):
            raise RejectionError("Invalid loaded V3 class order/threshold")
        self._verify()

    def _verify(self):
        if (self._runner.fit_count != self._generation or self._runner.model_identity(self._model.fitted) != self._identity
                or self._runner.fingerprints(self._model.fitted) != self._expected):
            raise RejectionError("Loaded recognizer fitted state changed")

    def predict_window(self, features):
        config = self._model.frozen_configuration
        X = np.asarray(features)
        if (X.shape != (config.temporal.window_size, config.pipeline.input_features)
                or X.dtype.kind not in "fi" or not np.isfinite(X).all()):
            raise RejectionError("Recognition requires exactly seven finite all_xyz_pose_v2 feature rows")
        self._verify()
        frames = self._runner.infer(self._model.fitted, V3Population(X, np.zeros(len(X), dtype=np.int32), ()))
        self._verify()
        if tuple(name for name, _ in frames.confidence) != (config.pipeline.novelty_method,):
            raise RejectionError("Unexpected novelty score family")
        scores = frames.confidence[0][1]
        if np.shape(scores) != (config.temporal.window_size,) or not np.isfinite(scores).all():
            raise RejectionError("Invalid window knownness scores")
        index, _ = vote_identity(frames.predicted, frames.decisions)
        score = float(np.median(scores))
        accepted = score >= self._model.threshold
        return V3RecognitionDecision(IdentityDecision.KNOWN if accepted else IdentityDecision.UNKNOWN,
            self._model.class_order[index] if accepted else None, score, self._model.threshold)
