"""Exact deterministic rejection calibration; no score arrays leave results."""

from collections.abc import Iterator
from dataclasses import dataclass
from math import isfinite, nextafter, inf

import numpy as np

from face_recognition.application.unknown_rejection_split import UnknownRejectionError
from face_recognition.domain.models.unknown_rejection import RejectionMetrics, ThresholdCalibration


def validated_decision_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or not len(scores) or scores.shape[1] < 2 or not np.isfinite(scores).all():
        raise UnknownRejectionError("Decision scores must be finite with at least two class columns")
    return scores


def decision_margin(scores: np.ndarray) -> np.ndarray:
    scores = validated_decision_scores(scores)
    top = np.partition(scores, -2, axis=1)[:, -2:]
    with np.errstate(over="ignore"):
        margin = top[:, 1] - top[:, 0]
    if not np.isfinite(margin).all():
        raise UnknownRejectionError("Decision margins must be finite")
    return margin


def _samples(known: np.ndarray, correct: np.ndarray, unknown: np.ndarray, *, allow_negative: bool = False):
    known, unknown, correct = np.asarray(known), np.asarray(unknown), np.asarray(correct)
    if (known.ndim != 1 or unknown.ndim != 1 or not len(known) or not len(unknown)
            or correct.shape != known.shape or correct.dtype != np.bool_
            or not np.isfinite(known).all() or not np.isfinite(unknown).all()
            or (not allow_negative and (np.any(known < 0) or np.any(unknown < 0)))):
        raise UnknownRejectionError("Invalid known/unknown confidence observations")
    return known, correct, unknown


def threshold_candidates(known: np.ndarray, unknown: np.ndarray, *, allow_negative: bool = False) -> np.ndarray:
    known, _, unknown = _samples(known, np.ones(np.shape(known), dtype=bool), unknown, allow_negative=allow_negative)
    values = np.unique(np.concatenate((known, unknown)).astype(np.float64, copy=False))
    upper = nextafter(float(values[-1]), inf)
    if not isfinite(upper):
        raise UnknownRejectionError("No finite reject-all threshold boundary is representable")
    with np.errstate(over="ignore"):
        midpoints = values[:-1] + (values[1:] - values[:-1]) / 2
    fallback = values[:-1] / 2 + values[1:] / 2
    midpoints = np.where(np.isfinite(midpoints), midpoints, fallback)
    midpoints = midpoints[(midpoints > values[:-1]) & (midpoints < values[1:])]
    lower = min(0.0, float(values[0])) if allow_negative else 0.0
    return np.unique(np.concatenate(([lower], values, midpoints, [upper])))


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


def evaluate_threshold(known: np.ndarray, correct: np.ndarray, unknown: np.ndarray, threshold: float,
                       *, allow_negative: bool = False) -> RejectionMetrics:
    known, correct, unknown = _samples(known, correct, unknown, allow_negative=allow_negative)
    if not isfinite(threshold) or (not allow_negative and threshold < 0):
        raise UnknownRejectionError("Threshold must be finite and obey the configured score direction")
    accepted = known >= threshold
    return _metrics(len(known), len(unknown), int((~accepted).sum()), int((unknown < threshold).sum()),
                    int((accepted & correct).sum()))


@dataclass(frozen=True)
class ThresholdPoint:
    threshold: float
    known_rejected: int
    unknown_rejected: int
    metrics: RejectionMetrics


def threshold_points(known: np.ndarray, correct: np.ndarray, unknown: np.ndarray,
                     *, allow_negative: bool = False) -> Iterator[ThresholdPoint]:
    """Stream exact operating-point counts/metrics without sample-wise rescans."""
    known, correct, unknown = _samples(known, correct, unknown, allow_negative=allow_negative)
    thresholds = threshold_candidates(known, unknown, allow_negative=allow_negative)
    order = np.argsort(known, kind="stable")
    rejected_correct = np.concatenate(([0], np.cumsum(correct[order], dtype=np.int64)))
    kr = np.searchsorted(known[order], thresholds, side="left")
    ur = np.searchsorted(np.sort(unknown), thresholds, side="left")
    nk, nu, total_correct = len(known), len(unknown), int(correct.sum())
    for threshold, rejected_known, rejected_unknown in zip(thresholds, kr, ur):
        rejected_known, rejected_unknown = int(rejected_known), int(rejected_unknown)
        metrics = _metrics(nk, nu, rejected_known, rejected_unknown,
                           total_correct - int(rejected_correct[rejected_known]))
        yield ThresholdPoint(float(threshold), rejected_known, rejected_unknown, metrics)


def calibrate_threshold(known: np.ndarray, correct: np.ndarray, unknown: np.ndarray,
                        *, allow_negative: bool = False) -> ThresholdCalibration:
    best_key, best, count = None, None, 0
    nk, nu = len(known), len(unknown)
    for point in threshold_points(known, correct, unknown, allow_negative=allow_negative):
        count += 1
        # Integer numerators compare balanced accuracy exactly, avoiding false
        # non-ties from float rounding. Denominators are fixed within this fold.
        key = ((nk - point.known_rejected) * nu + point.unknown_rejected * nk,
               -(nu - point.unknown_rejected), -point.known_rejected, point.threshold)
        if best_key is None or key > best_key:
            best_key, best = key, point
    return ThresholdCalibration(best.threshold, best.metrics, count)
