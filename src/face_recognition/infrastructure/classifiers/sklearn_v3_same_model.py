"""Fit-generation and in-memory state fingerprints for same-model scoring.

No fitted artifact is persisted. Streaming pickle bytes feed SHA-256 directly;
hashes establish local state integrity, not authenticity or a model format.
"""

from hashlib import sha256
import pickle

from face_recognition.infrastructure.classifiers.sklearn_v3_novelty import SklearnV3NoveltyRunner


class _DigestSink:
    def __init__(self):
        self.digest = sha256()

    def write(self, content):
        self.digest.update(content)
        return memoryview(content).nbytes


class SklearnV3SameModelRunner(SklearnV3NoveltyRunner):
    def __init__(self):
        super().__init__()
        self.fit_count = 0

    def fit(self, population, identity_count, methods=("mahalanobis_distance",)):
        if methods != ("mahalanobis_distance",):
            raise ValueError("This calibration study fits Mahalanobis only")
        self.fit_count += 1
        return super().fit(population, identity_count, methods)

    @staticmethod
    def model_state_sha256(model):
        sink = _DigestSink()
        pickle.dump(model, sink, protocol=5)
        return sink.digest.hexdigest()

    @staticmethod
    def model_identity(model):
        return (id(model), id(model.pipeline), id(model.pipeline.named_steps["anova"]),
                id(model.pipeline.named_steps["scaler"]), id(model.pipeline[-1]), id(model.mahalanobis))
