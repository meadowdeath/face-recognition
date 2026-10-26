"""Focused V2 decision rules; score arrays are transient, results are aggregates."""

from dataclasses import asdict

import numpy as np

from face_recognition.application.rejection_metrics import _metrics, threshold_candidates
from face_recognition.application.rejection_score_analysis import score_distribution
from face_recognition.application.v2_study_data import V2StudyError
from face_recognition.domain.models.v2_diagnostics import (
    ConfusionSummary, DistributionSummary, IdentityThreshold, MIN_IDENTITY_SUPPORT,
    StrategyConfiguration, V2_STRATEGIES,
)


def configurations():
    return tuple(StrategyConfiguration(name, score, per_identity, size, gate)
                 for name, score, per_identity in V2_STRATEGIES
                 for size in (1, 3, 5, 7) for gate in ((False,) if size == 1 else (False, True)))


def gated_points(known, correct, unknown, known_gate, unknown_gate):
    """Exact existing candidate policy, including forced rejections in denominators.

    Sorted eligible observations avoid rescanning every sample per threshold.
    """
    known, unknown = np.asarray(known), np.asarray(unknown)
    correct, known_gate, unknown_gate = map(np.asarray, (correct, known_gate, unknown_gate))
    if (correct.shape != known.shape or correct.dtype != bool or known_gate.shape != known.shape
            or unknown_gate.shape != unknown.shape or known_gate.dtype != bool or unknown_gate.dtype != bool):
        raise V2StudyError("Invalid calibration decision masks")
    thresholds = threshold_candidates(known, unknown, allow_negative=True)
    eligible = np.flatnonzero(known_gate)
    order = eligible[np.argsort(known[eligible], kind="stable")]
    prefix = np.concatenate(([0], np.cumsum(correct[order], dtype=np.int64)))
    rejected = np.searchsorted(known[order], thresholds, side="left")
    unknown_below = np.searchsorted(np.sort(unknown[unknown_gate]), thresholds, side="left")
    nk, nu = len(known), len(unknown)
    for t, r, u in zip(thresholds, rejected, unknown_below):
        ka, ua = len(order) - int(r), int(unknown_gate.sum()) - int(u)
        yield float(t), ka, ua, _metrics(nk, nu, nk - ka, nu - ua, int(prefix[-1] - prefix[r]))


def calibrate(known, correct, unknown, known_gate, unknown_gate):
    nk, nu = len(known), len(unknown)
    return max(gated_points(known, correct, unknown, known_gate, unknown_gate),
               key=lambda p: (p[1] * nu + (nu - p[2]) * nk, -p[2], p[1], p[0]))


def identity_thresholds(ids, known, correct, unknown, kp, up, kg, ug, global_threshold):
    rows = []
    for label, person in enumerate(ids):
        k, u = kp == label, up == label
        ks, us = int((k & kg).sum()), int((u & ug).sum())
        available = ks >= MIN_IDENTITY_SUPPORT and us >= MIN_IDENTITY_SUPPORT
        threshold = calibrate(known[k], correct[k], unknown[u], kg[k], ug[u])[0] if available else None
        rows.append(IdentityThreshold(person, int(k.sum()), int(u.sum()), ks, us, threshold,
                                      threshold if available else global_threshold, available))
    return tuple(rows)


def far_points(known, correct, unknown, kg, ug):
    best = {target: None for target in (20, 10, 5)}
    for p in gated_points(known, correct, unknown, kg, ug):
        for target in best:
            if p[2] * 100 <= target * len(unknown):
                old = best[target]
                if old is None or (p[1], -p[2], p[0]) > (old[1], -old[2], old[0]):
                    best[target] = p
    return tuple({"far_target": target / 100, "status": "impractical" if p[1] * 100 <= len(known) else "available",
                  "threshold": p[0], "calibration_metrics": asdict(p[3])} for target, p in best.items())


def open_matrix(truth, predicted, accepted, unknown_predicted, unknown_accepted, identities):
    matrix = np.zeros((identities + 1, identities + 1), dtype=np.int64)
    np.add.at(matrix, (truth, np.where(accepted, predicted, identities)), 1)
    np.add.at(matrix, (np.full(len(unknown_predicted), identities),
                      np.where(unknown_accepted, unknown_predicted, identities)), 1)
    return matrix


def matrix_metrics(matrix):
    nk, nu = float(matrix[:-1].sum()), float(matrix[-1].sum())
    if not nk or not nu:
        raise V2StudyError("Both known and unknown evaluation populations are required")
    return _metrics(nk, nu, float(matrix[:-1, -1].sum()), float(matrix[-1, -1]),
                    float(np.trace(matrix[:-1, :-1])))


def confusion(matrix, labels, known_repetitions=1):
    adjusted = matrix.astype(np.float64)
    adjusted[:-1] /= known_repetitions
    rates = np.divide(matrix, matrix.sum(axis=1, keepdims=True), out=np.zeros(matrix.shape),
                      where=matrix.sum(axis=1, keepdims=True) != 0)
    return ConfusionSummary(tuple(labels), tuple(tuple(int(v) for v in row) for row in matrix),
                            tuple(map(tuple, rates.tolist())), tuple(map(tuple, adjusted.tolist())))


def binary_matrix(matrix):
    return np.array([[matrix[:-1, :-1].sum(), matrix[:-1, -1].sum()],
                     [matrix[-1, :-1].sum(), matrix[-1, -1]]], dtype=np.int64)


def distribution(values):
    return DistributionSummary(len(values), score_distribution(values) if len(values) else None)


def rotation_masks(people):
    people = np.asarray(people)
    return tuple((people != person, people == person) for person in sorted(set(people.tolist())))


def rank_key(row):
    m, c = row["metrics"], row["configuration"]
    return (-row["mean_roc_auc"], row["mean_eer"], m["unknown_false_acceptance_rate"],
            m["known_rejection_rate"], -m["known_correct_identification_rate"],
            -m["open_set_exact_accuracy"], m["known_misidentification_rate"],
            c["per_identity"], c["strict_majority_gate"], c["window"], c["strategy"])
