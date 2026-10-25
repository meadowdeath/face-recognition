"""Continuous-score separation and exact diagnostic operating points."""

import numpy as np

from face_recognition.application.rejection_metrics import calibrate_threshold, threshold_points
from face_recognition.application.unknown_rejection_split import UnknownRejectionError
from face_recognition.domain.models.rejection_score_evaluation import (
    FAR_TARGETS, EqualErrorPoint, FarOperatingPoint, ScoreAnalysis, ScoreDistribution,
)


def _scores(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise UnknownRejectionError("Confidence scores must be a finite nonempty vector")
    return values


def score_distribution(values: np.ndarray) -> ScoreDistribution:
    values = _scores(values)
    p10, p25, median, p75, p90 = np.percentile(values, (10, 25, 50, 75, 90), method="linear")
    return ScoreDistribution(len(values), float(np.mean(values)), float(np.std(values, ddof=0)),
                              float(median), float(p10), float(p25), float(p75), float(p90))


def roc_auc_known_positive(known: np.ndarray, unknown: np.ndarray) -> float:
    """Mann-Whitney AUC: P(known>unknown) + 0.5*P(tie), no binarization."""
    known, unknown = _scores(known), np.sort(_scores(unknown))
    below = np.searchsorted(unknown, known, side="left")
    through = np.searchsorted(unknown, known, side="right")
    return (2 * int(below.sum()) + int((through - below).sum())) / (2 * len(known) * len(unknown))


def analyze_scores(known: np.ndarray, correct: np.ndarray, unknown: np.ndarray) -> ScoreAnalysis:
    calibration = calibrate_threshold(known, correct, unknown, allow_negative=True)
    nk, nu = len(known), len(unknown)
    eer_key, eer = None, None
    far_best = {target: None for target in FAR_TARGETS}
    far_keys = {target: None for target in FAR_TARGETS}
    for point in threshold_points(known, correct, unknown, allow_negative=True):
        ua = nu - point.unknown_rejected
        ka = nk - point.known_rejected
        # Common denominator nk*nu makes exact closest-error comparisons possible.
        key = (abs(ua * nk - point.known_rejected * nu), ua, point.known_rejected, -point.threshold)
        if eer_key is None or key < eer_key:
            eer_key = key
            metrics = point.metrics
            eer = EqualErrorPoint((metrics.unknown_false_acceptance_rate + metrics.known_rejection_rate) / 2,
                                  point.threshold, metrics.unknown_false_acceptance_rate, metrics.known_rejection_rate)
        for target in FAR_TARGETS:
            if ua * 100 <= int(round(target * 100)) * nu:
                far_key = (ka, -ua, point.threshold)
                if far_keys[target] is None or far_key > far_keys[target]:
                    far_keys[target], far_best[target] = far_key, point
    operating_points = []
    for target in FAR_TARGETS:
        point = far_best[target]
        if point is None:
            operating_points.append(FarOperatingPoint(target, "unavailable", None, None, "No finite candidate satisfies FAR"))
        else:
            impractical = 100 * (nk - point.known_rejected) <= nk
            operating_points.append(FarOperatingPoint(target, "impractical" if impractical else "available",
                                                      point.threshold, point.metrics,
                                                      "At most 1% of known samples accepted" if impractical else None))
    return ScoreAnalysis(roc_auc_known_positive(known, unknown), eer, calibration, tuple(operating_points),
                          score_distribution(known), score_distribution(unknown))
