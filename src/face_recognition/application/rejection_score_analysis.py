"""Continuous-score separation and exact diagnostic operating points."""

import numpy as np

from face_recognition.application.rejection_metrics import threshold_points
from face_recognition.domain.models.rejection import RejectionError
from face_recognition.domain.models.score_analysis import (
    EqualErrorPoint, ScoreDistribution, ScoreSeparation,
)


def _scores(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise RejectionError("Confidence scores must be a finite nonempty vector")
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


def _eer_key(point, nk: int, nu: int):
    return (abs((nu - point.unknown_rejected) * nk - point.known_rejected * nu),
            nu - point.unknown_rejected, point.known_rejected, -point.threshold)


def score_separation(known: np.ndarray, unknown: np.ndarray) -> ScoreSeparation:
    """ROC/EER diagnostics, NEVER calls calibration or chooses operational threshold."""
    known, unknown = _scores(known), _scores(unknown)
    point = min(threshold_points(known, np.ones(len(known), dtype=bool), unknown, allow_negative=True),
                key=lambda row: _eer_key(row, len(known), len(unknown)))
    m = point.metrics
    eer = EqualErrorPoint((m.unknown_false_acceptance_rate + m.known_rejection_rate) / 2, point.threshold,
                          m.unknown_false_acceptance_rate, m.known_rejection_rate)
    return ScoreSeparation(roc_auc_known_positive(known, unknown), eer, score_distribution(known), score_distribution(unknown))
