"""Small fit/score primitives; centroids and scales never leave the runtime."""

import numpy as np

from face_recognition.application.rejection_metrics import validated_decision_scores
from face_recognition.application.unknown_rejection_split import UnknownRejectionError
from face_recognition.domain.models.rejection_score_evaluation import CENTROID_SCALE_EPSILON


def decision_top1(scores: np.ndarray) -> np.ndarray:
    return validated_decision_scores(scores).max(axis=1)


class PredictedClassCentroidScorer:
    def fit(self, X_fit: np.ndarray, y_fit: np.ndarray, identity_count: int) -> "PredictedClassCentroidScorer":
        X_fit, y_fit = np.asarray(X_fit), np.asarray(y_fit)
        if (X_fit.ndim != 2 or not len(X_fit) or not X_fit.shape[1] or not np.isfinite(X_fit).all()
                or y_fit.shape != (len(X_fit),) or not np.issubdtype(y_fit.dtype, np.integer)
                or identity_count < 2 or not np.array_equal(np.unique(y_fit), np.arange(identity_count))):
            raise UnknownRejectionError("Invalid centroid fit data or class ordering")
        centers, scales = [], []
        for label in range(identity_count):
            samples = X_fit[y_fit == label]
            center = samples.mean(axis=0, dtype=np.float64)
            distances = np.linalg.norm(samples - center, axis=1)
            centers.append(center)
            scales.append(max(float(np.median(distances)), CENTROID_SCALE_EPSILON))
        self._centroids = np.asarray(centers)
        self._scales = np.asarray(scales)
        if not np.isfinite(self._centroids).all() or not np.isfinite(self._scales).all():
            raise UnknownRejectionError("Nonfinite centroid reference statistics")
        self._centroids.setflags(write=False)
        self._scales.setflags(write=False)
        self.fit_samples = len(X_fit)
        return self

    def score_pair(self, X: np.ndarray, predicted_class: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not hasattr(self, "_centroids"):
            raise UnknownRejectionError("Centroid scorer must be fitted first")
        X, labels = np.asarray(X), np.asarray(predicted_class)
        if (X.ndim != 2 or X.shape[1] != self._centroids.shape[1] or not len(X) or not np.isfinite(X).all()
                or labels.shape != (len(X),) or not np.issubdtype(labels.dtype, np.integer)
                or np.any(labels < 0) or np.any(labels >= len(self._centroids))):
            raise UnknownRejectionError("Invalid centroid score data or predicted class metadata")
        # Advanced indexing creates one work matrix. Subtract in-place to avoid
        # additional full-size distance/square matrices.
        delta = self._centroids[labels]
        np.subtract(X, delta, out=delta)
        distance = np.sqrt(np.einsum("ij,ij->i", delta, delta))
        normalized = distance / self._scales[labels]
        if not np.isfinite(distance).all() or not np.isfinite(normalized).all():
            raise UnknownRejectionError("Centroid confidence must be finite")
        return -distance, -normalized

    def score(self, X: np.ndarray, predicted_class: np.ndarray, *, normalized: bool = False) -> np.ndarray:
        return self.score_pair(X, predicted_class)[int(normalized)]

    def __repr__(self) -> str:
        return f"PredictedClassCentroidScorer(fitted={hasattr(self, '_centroids')})"
