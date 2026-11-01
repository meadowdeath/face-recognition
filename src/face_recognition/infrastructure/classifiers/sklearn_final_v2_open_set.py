"""Reuse persisted V2 top1 inference only; no training or threshold selection."""

from dataclasses import replace

from face_recognition.infrastructure.classifiers.sklearn_final_open_set import FinalInference
from face_recognition.infrastructure.classifiers.sklearn_v2_diagnostics import identity_summary
from face_recognition.infrastructure.classifiers.sklearn_v2_open_set_refit import SklearnV2OpenSetRefitRunner


class SklearnFinalV2OpenSetRunner:
    def __init__(self, model):
        self.model = model
        self.inference = SklearnV2OpenSetRefitRunner()
        self.versions = self.inference.versions
        self.frozen_threshold = model.threshold
        self.class_ids = model.parts.class_ids

    def infer(self, known_X, known_y, unknown_X):
        known = self.inference.infer(self.model.parts, known_X)
        unknown = self.inference.infer(self.model.parts, unknown_X)
        closed = replace(identity_summary(known_y, known.predicted, self.class_ids),
                         scope="Independent final V2 KNOWN frames before rejection, including temporal trailing discards")
        return FinalInference(known, unknown, closed)
