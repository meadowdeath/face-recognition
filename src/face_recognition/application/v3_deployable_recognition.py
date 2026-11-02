"""Window-level inference only. UNKNOWN never emits an arbitrary registered ID."""

from math import isfinite

import numpy as np

from face_recognition.application.temporal_windows import vote_identity
from face_recognition.application.v3_novelty_data import V3Population
from face_recognition.domain.models.rejection import RejectionError
from face_recognition.domain.models.v3_deployable_model import (
    IdentityDecision, V3FrameObservation, V3RecognitionDecision, V3TemporalOutcome,
)


class V3DeployableRecognizer:
    """One coherent seven-frame feature window; callers own capture boundaries.

    The threshold is calibrated for windows, so there is deliberately no
    operational single-frame threshold API. infer_observation exposes candidate
    evidence only; decide_observations requires exactly seven such observations.
    """
    def __init__(self, model, runner, expected_fingerprints):
        self._model, self._runner = model, runner
        self._loaded_model, self._loaded_runner = model, runner
        self._fitted = model.fitted
        self._configuration = model.frozen_configuration
        self._pipeline_config, self._temporal_config = self._configuration.pipeline, self._configuration.temporal
        self._classes = model.class_order
        self._expected = expected_fingerprints
        self._generation = runner.fit_count
        self._identity = runner.model_identity(model.fitted)
        self._contract = (model.threshold, model.fitted.identity_count, self.representation,
                          self.input_width, self.window_size, self._pipeline_config.novelty_method)
        if len(model.class_order) != model.fitted.identity_count or not isfinite(model.threshold):
            raise RejectionError("Invalid loaded V3 class order/threshold")
        self.verify_integrity()  # Startup only; loader also validates files and full fitted state.

    def _verify(self):
        """Constant-size runtime guards, never serialize/hash learned arrays."""
        if (self._model is not self._loaded_model or self._runner is not self._loaded_runner
                or self._model.fitted is not self._fitted or self._model.frozen_configuration is not self._configuration
                or self._configuration.pipeline is not self._pipeline_config or self._configuration.temporal is not self._temporal_config
                or self._model.class_order is not self._classes or self._runner.fit_count != self._generation
                or self._runner.model_identity(self._model.fitted) != self._identity
                or (self._model.threshold, self._fitted.identity_count, self.representation,
                    self.input_width, self.window_size, self._pipeline_config.novelty_method) != self._contract
                or not isfinite(self._model.threshold)):
            raise RejectionError("Loaded recognizer fitted state changed")

    def verify_integrity(self):
        """Explicit expensive diagnostic; never called by normal inference.

        Startup/load checks detect altered learned values. Runtime identity and
        generation guards detect replacement/refitting, not arbitrary in-place
        array edits; callers can request this full check outside the hot loop.
        """
        self._verify()
        if self._runner.fingerprints(self._model.fitted) != self._expected:
            raise RejectionError("Loaded recognizer fitted state changed")

    @property
    def representation(self):
        return self._model.frozen_configuration.pipeline.representation

    @property
    def input_width(self):
        return self._model.frozen_configuration.pipeline.input_features

    @property
    def window_size(self):
        return self._model.frozen_configuration.temporal.window_size

    @property
    def threshold(self):
        return self._model.threshold

    def _infer_observations(self, X):
        self._verify()
        frames = self._runner.infer(self._model.fitted, V3Population(X, np.zeros(len(X), dtype=np.int32), ()))
        self._verify()
        method = self._model.frozen_configuration.pipeline.novelty_method
        if tuple(name for name, _ in frames.confidence) != (method,):
            raise RejectionError("Unexpected novelty score family")
        scores = frames.confidence[0][1]
        classes = len(self._model.class_order)
        if (np.shape(scores) != (len(X),) or not np.isfinite(scores).all()
                or frames.predicted.shape != (len(X),) or not np.issubdtype(frames.predicted.dtype, np.integer)
                or np.any(frames.predicted < 0) or np.any(frames.predicted >= classes)
                or frames.decisions.shape != (len(X), classes) or not np.isfinite(frames.decisions).all()):
            raise RejectionError("Invalid candidate inference outputs")
        return tuple(V3FrameObservation(int(index), self._model.class_order[int(index)], float(score), tuple(float(v) for v in decisions))
                     for index, score, decisions in zip(frames.predicted, scores, frames.decisions))

    def infer_observation(self, features):
        """Score one face. This does not apply the window-calibrated threshold."""
        X = np.asarray(features)
        if X.shape != (self.input_width,) or X.dtype.kind not in "fi" or not np.isfinite(X).all():
            raise RejectionError("A face observation requires 1434 finite all_xyz_pose_v2 features")
        return self._infer_observations(X.reshape(1, -1))[0]

    def decide_observations(self, observations):
        """Reuse the exact frozen plurality, score-sum/class-order and median rules."""
        if len(observations) != self.window_size:
            raise RejectionError("A temporal decision requires exactly seven successful observations")
        self._verify()
        for row in observations:
            if (type(row.candidate_index) is not int or not 0 <= row.candidate_index < len(self._model.class_order)
                    or row.candidate_identity != self._model.class_order[row.candidate_index]
                    or len(row.decision_scores) != len(self._model.class_order) or not isfinite(row.knownness_score)):
                raise RejectionError("Invalid temporal candidate evidence")
        predicted = np.array([r.candidate_index for r in observations], dtype=np.int32)
        decisions = np.array([r.decision_scores for r in observations], dtype=np.float64)
        index, _ = vote_identity(predicted, decisions)
        score = float(np.median([r.knownness_score for r in observations]))
        accepted = score >= self.threshold
        decision = V3RecognitionDecision(IdentityDecision.KNOWN if accepted else IdentityDecision.UNKNOWN,
            self._model.class_order[index] if accepted else None, score, self.threshold)
        return V3TemporalOutcome(decision, self._model.class_order[index])

    def predict_window(self, features):
        config = self._model.frozen_configuration
        X = np.asarray(features)
        if (X.shape != (config.temporal.window_size, config.pipeline.input_features)
                or X.dtype.kind not in "fi" or not np.isfinite(X).all()):
            raise RejectionError("Recognition requires exactly seven finite all_xyz_pose_v2 feature rows")
        return self.decide_observations(self._infer_observations(X)).decision
