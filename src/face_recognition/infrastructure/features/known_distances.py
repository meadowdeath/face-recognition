"""Batched, population-wide Euclidean novelty scores, not identity classification."""

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

from face_recognition.application.unknown_rejection_split import UnknownRejectionError
from face_recognition.domain.models.distance_novelty import NOVELTY_NEIGHBORS


@dataclass(frozen=True, eq=False)
class KnownDistanceObservations:
    confidence: tuple[tuple[str, np.ndarray], ...] = field(repr=False)
    nearest_labels: np.ndarray = field(repr=False)


class KnownDistanceScorer:
    """Own one float64 reference copy; use <=batch_size by N distance work arrays.

    All three scores share the same five nearest distances. Row ties only
    affect the aggregate identity diagnostic, never SVM identity predictions.
    """

    def __init__(self, *, batch_size: int = 128) -> None:
        if type(batch_size) is not int or not 1 <= batch_size <= 128:
            raise UnknownRejectionError("Distance batch size must be an integer from 1 to 128")
        self.batch_size = batch_size

    @staticmethod
    def _matrix(X: np.ndarray, *, copy: bool = False) -> np.ndarray:
        try:
            values = np.array(X, dtype=np.float64, copy=True) if copy else np.asarray(X, dtype=np.float64)
            if values.ndim != 2 or not len(values) or not values.shape[1] or not np.isfinite(values).all():
                raise ValueError
            return values
        except (ValueError, TypeError, OverflowError):
            raise UnknownRejectionError("Distance inputs must be a finite nonempty numeric matrix") from None

    def fit(self, X_fit: np.ndarray, y_fit: np.ndarray, identity_count: int) -> "KnownDistanceScorer":
        values, labels = self._matrix(X_fit, copy=True), np.asarray(y_fit)
        if len(values) < 5:
            raise UnknownRejectionError("At least 5 known fit reference samples are required for the 1/3/5-neighbor experiment")
        if (labels.shape != (len(values),) or not np.issubdtype(labels.dtype, np.integer)
                or type(identity_count) is not int or identity_count < 2
                or not np.array_equal(np.unique(labels), np.arange(identity_count))):
            raise UnknownRejectionError("Distance references require deterministic known fit labels")
        self._reference, self._labels = values, labels.copy()
        self._reference.setflags(write=False)
        self._labels.setflags(write=False)
        self.fit_samples = len(values)
        return self

    def score(self, X: np.ndarray) -> KnownDistanceObservations:
        if not hasattr(self, "_reference"):
            raise UnknownRejectionError("Known distance scorer must be fitted first")
        # Conversion is batched too; avoid a full float64 query-matrix copy.
        X = np.asarray(X)
        if X.ndim != 2 or not len(X) or X.shape[1] != self._reference.shape[1]:
            raise UnknownRejectionError("Distance query dimensions must match known references")
        confidence = [(name, np.empty(len(X), dtype=np.float64)) for name, _ in NOVELTY_NEIGHBORS]
        nearest_labels = np.empty(len(X), dtype=self._labels.dtype)
        for start in range(0, len(X), self.batch_size):
            stop = min(start + self.batch_size, len(X))
            query = self._matrix(X[start:stop])
            with np.errstate(over="ignore", invalid="ignore"):
                distances = euclidean_distances(query, self._reference)
            if not np.isfinite(distances).all():
                raise UnknownRejectionError("Nonfinite Euclidean distance calculation")
            nearest_labels[start:stop] = self._labels[np.argmin(distances, axis=1)]
            distances.partition(4, axis=1)
            five = np.sort(distances[:, :5], axis=1)
            for (_, count), (_, output) in zip(NOVELTY_NEIGHBORS, confidence):
                output[start:stop] = -five[:, :count].mean(axis=1)
        return KnownDistanceObservations(tuple(confidence), nearest_labels)

    def __repr__(self) -> str:
        return f"KnownDistanceScorer(fitted={hasattr(self, '_reference')}, batch_size={self.batch_size})"
