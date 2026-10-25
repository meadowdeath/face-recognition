"""Exact deterministic rejection calibration; no score arrays leave results."""

from math import isfinite, nextafter, inf

import numpy as np

from face_recognition.application.unknown_rejection_split import UnknownRejectionError
from face_recognition.domain.models.unknown_rejection import RejectionMetrics, ThresholdCalibration


def decision_margin(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or not len(scores) or scores.shape[1] < 2 or not np.isfinite(scores).all():
        raise UnknownRejectionError("Decision scores must be finite with at least two class columns")
    top = np.partition(scores, -2, axis=1)[:, -2:]
    with np.errstate(over="ignore"):
        margin = top[:, 1] - top[:, 0]
    if not np.isfinite(margin).all():
        raise UnknownRejectionError("Decision margins must be finite")
    return margin


def _samples(known: np.ndarray, correct: np.ndarray, unknown: np.ndarray):
    known, unknown, correct = np.asarray(known), np.asarray(unknown), np.asarray(correct)
    if (known.ndim != 1 or unknown.ndim != 1 or not len(known) or not len(unknown)
            or correct.shape != known.shape or correct.dtype != np.bool_
            or not np.isfinite(known).all() or not np.isfinite(unknown).all()
            or np.any(known < 0) or np.any(unknown < 0)):
        raise UnknownRejectionError("Invalid known/unknown confidence observations")
    return known, correct, unknown


def threshold_candidates(known: np.ndarray, unknown: np.ndarray) -> np.ndarray:
    known, _, unknown = _samples(known, np.ones(np.shape(known), dtype=bool), unknown)
    values = np.unique(np.concatenate((known, unknown)).astype(np.float64, copy=False))
    upper = nextafter(float(values[-1]), inf)
    if not isfinite(upper):
        raise UnknownRejectionError("No finite reject-all threshold boundary is representable")
    midpoints = values[:-1] + (values[1:] - values[:-1]) / 2
    midpoints = midpoints[(midpoints > values[:-1]) & (midpoints < values[1:])]
    return np.unique(np.concatenate(([0.0], values, midpoints, [upper])))


def _metrics(nk: int, nu: int, known_rejected: int, unknown_rejected: int, correct_accepted: int) -> RejectionMetrics:
    ka, ua = nk - known_rejected, nu - unknown_rejected
    known_f1_denominator = 2 * ka + ua + known_rejected
    unknown_f1_denominator = 2 * unknown_rejected + known_rejected + ua
    known_f1 = 2 * ka / known_f1_denominator if known_f1_denominator else 0.0
    unknown_f1 = 2 * unknown_rejected / unknown_f1_denominator if unknown_f1_denominator else 0.0
    return RejectionMetrics(ka / nk, known_rejected / nk, unknown_rejected / nu, ua / nu,
                            (ka / nk + unknown_rejected / nu) / 2, (known_f1 + unknown_f1) / 2,
                            correct_accepted / nk, (ka - correct_accepted) / nk,
                            (correct_accepted + unknown_rejected) / (nk + nu))


def evaluate_threshold(known: np.ndarray, correct: np.ndarray, unknown: np.ndarray, threshold: float) -> RejectionMetrics:
    known, correct, unknown = _samples(known, correct, unknown)
    if not isfinite(threshold) or threshold < 0:
        raise UnknownRejectionError("Threshold must be finite and nonnegative")
    accepted = known >= threshold
    return _metrics(len(known), len(unknown), int((~accepted).sum()), int((unknown < threshold).sum()),
                    int((accepted & correct).sum()))


def calibrate_threshold(known: np.ndarray, correct: np.ndarray, unknown: np.ndarray) -> ThresholdCalibration:
    known, correct, unknown = _samples(known, correct, unknown)
    thresholds = threshold_candidates(known, unknown)
    order = np.argsort(known, kind="stable")
    rejected_correct = np.concatenate(([0], np.cumsum(correct[order], dtype=np.int64)))
    kr = np.searchsorted(known[order], thresholds, side="left")
    ur = np.searchsorted(np.sort(unknown), thresholds, side="left")
    nk, nu, total_correct = len(known), len(unknown), int(correct.sum())
    best_key, best = None, None
    for threshold, rejected_known, rejected_unknown in zip(thresholds, kr, ur):
        rejected_known, rejected_unknown = int(rejected_known), int(rejected_unknown)
        # Integer numerators compare balanced accuracy exactly, avoiding false
        # non-ties from float rounding. Denominators are fixed within this fold.
        key = ((nk - rejected_known) * nu + rejected_unknown * nk,
               -(nu - rejected_unknown), -rejected_known, float(threshold))
        metrics = _metrics(nk, nu, rejected_known, rejected_unknown,
                           total_correct - int(rejected_correct[rejected_known]))
        if best_key is None or key > best_key:
            best_key, best = key, ThresholdCalibration(float(threshold), metrics, len(thresholds))
    return best
