"""Fit-only per-class shrinkage distributions, evaluated at the SVM prediction."""

import numpy as np
from scipy.linalg import solve_triangular
from sklearn.covariance import LedoitWolf

from face_recognition.domain.models.rejection import RejectionError


class PredictedClassMahalanobisScorer:
    def fit(self, X_fit, y_fit, identity_count):
        X, y = np.asarray(X_fit), np.asarray(y_fit)
        if (X.ndim != 2 or not X.shape[1] or not np.isfinite(X).all() or y.shape != (len(X),)
                or not np.issubdtype(y.dtype, np.integer) or identity_count < 2
                or not np.array_equal(np.unique(y), np.arange(identity_count))):
            raise RejectionError("Invalid Mahalanobis fit population")
        centers, factors = [], []
        for label in range(identity_count):
            rows = np.asarray(X[y == label], dtype=np.float64)
            if len(rows) < 2:
                raise RejectionError("Shrinkage references require at least two fit samples per identity")
            estimate = LedoitWolf(store_precision=False, assume_centered=False).fit(rows)
            covariance = estimate.covariance_.copy()
            # Positive ridge also covers a completely constant class, for which
            # shrinkage alone yields a zero covariance and zero precision.
            ridge = 1e-8 * max(1., float(np.trace(covariance)) / X.shape[1])
            covariance.flat[::X.shape[1] + 1] += ridge
            factor = np.linalg.cholesky(covariance)
            if not np.isfinite(factor).all() or not np.isfinite(estimate.location_).all():
                raise RejectionError("Nonfinite shrinkage distribution")
            centers.append(estimate.location_)
            factors.append(factor)
        self._centers, self._factors = np.asarray(centers), tuple(factors)
        self.fit_samples = len(X)
        return self

    def score(self, X, predicted_class):
        if not hasattr(self, "_centers"):
            raise RejectionError("Mahalanobis scorer must be fitted first")
        X, predicted = np.asarray(X), np.asarray(predicted_class)
        if (X.ndim != 2 or not len(X) or X.shape[1] != self._centers.shape[1] or not np.isfinite(X).all()
                or predicted.shape != (len(X),) or not np.issubdtype(predicted.dtype, np.integer)
                or np.any(predicted < 0) or np.any(predicted >= len(self._centers))):
            raise RejectionError("Invalid Mahalanobis query metadata")
        result = np.empty(len(X), dtype=np.float64)
        for label in np.unique(predicted):
            mask = predicted == label
            delta = np.asarray(X[mask], dtype=np.float64) - self._centers[label]
            whitened = solve_triangular(self._factors[label], delta.T, lower=True, check_finite=False)
            result[mask] = -np.sqrt(np.einsum("ij,ij->j", whitened, whitened))
        if not np.isfinite(result).all():
            raise RejectionError("Nonfinite Mahalanobis distance")
        return result

    def __repr__(self):
        return f"PredictedClassMahalanobisScorer(fitted={hasattr(self, '_centers')})"
