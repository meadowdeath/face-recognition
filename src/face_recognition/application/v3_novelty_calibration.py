"""Calibration-only pooling, declared ranking and descriptive threshold transport."""

from dataclasses import fields
from statistics import fmean, pstdev

import numpy as np

from face_recognition.application.rejection_metrics import evaluate_threshold
from face_recognition.application.rejection_score_analysis import score_distribution, score_separation
from face_recognition.domain.models.rejection import RejectionError, RejectionMetrics
from face_recognition.domain.models.v3_novelty_evaluation import (
    CANDIDATES, BinaryCalibrationMetrics, CandidateSummary, PoolCounts,
    ThresholdTransport, TransferPopulationDiagnostics,
)


def moments(rows, model):
    rows = tuple(rows)
    return tuple(model(*(operation(getattr(row, f.name) for row in rows) for f in fields(model)))
                 for operation in (fmean, pstdev))


def summarize_candidates(results):
    summaries = []
    for method in CANDIDATES:
        rows = tuple(r for r in results if r.method == method)
        mean, std = moments((r.validation_metrics for r in rows), RejectionMetrics)
        auc = [r.validation_diagnostics.roc_auc for r in rows]
        eer = [r.validation_diagnostics.equal_error_point.eer for r in rows]
        summaries.append(CandidateSummary(method, len(rows), mean, std, fmean(auc), pstdev(auc), fmean(eer), pstdev(eer)))
    return tuple(summaries)


def candidate_ranking(summaries):
    if tuple(s.method for s in summaries) != CANDIDATES:
        raise RejectionError("Require the four declared candidates in fixed order")
    return tuple(s.method for s in sorted(summaries, key=lambda s: (
        -s.mean_metrics.binary_balanced_accuracy, s.mean_metrics.unknown_false_acceptance_rate,
        s.mean_metrics.known_rejection_rate, -s.mean_roc_auc, CANDIDATES.index(s.method))))


def evaluate_pair(known, unknown, threshold):
    return (evaluate_threshold(known.scores, known.correct, unknown.scores, threshold, allow_negative=True),
            score_separation(known.scores, unknown.scores))


def binary_metrics(metrics):
    # Pooled cross-model confidence has no single identity prediction. Report
    # binary calibration only, not an invented cross-model identity vote.
    return BinaryCalibrationMetrics(*(getattr(metrics, f.name) for f in fields(BinaryCalibrationMetrics)))


class CrossFittedWindowPool:
    """Ephemeral scores keyed by physical window, then distinct fit population.

    Callers add INNER CALIBRATION batches only. Repeated UNKNOWN rotations
    contribute identical scores under a given fit model and count once. Median
    across distinct models follows window-7 median aggregation, not vice versa.
    No keys or individual scores are serialized in reports.
    """
    def __init__(self):
        self._rows = {}
        self._raw_count = 0

    def add(self, model_key, batch):
        if len(batch.keys) != len(batch.scores) or not np.isfinite(batch.scores).all():
            raise RejectionError("Invalid cross-fitted calibration observations")
        for key, value in zip(batch.keys, batch.scores):
            models = self._rows.setdefault(key, {})
            if model_key in models and models[model_key] != float(value):
                raise RejectionError("Same fit population produced inconsistent repeated calibration score")
            models[model_key] = float(value)
            self._raw_count += 1

    def finish(self, expected_models):
        if not self._rows or any(len(models) != expected_models for models in self._rows.values()):
            raise RejectionError("Incomplete cross-fitted physical-window coverage")
        scores = np.array([np.median(tuple(self._rows[key].values())) for key in sorted(self._rows)], dtype=np.float64)
        counts = PoolCounts(self._raw_count, sum(map(len, self._rows.values())), len(self._rows), expected_models, expected_models)
        return scores, counts


def transport_diagnostics(cross_known, cross_unknown, final_known, final_unknown, threshold):
    """No alignment, retuning or failover. Outer values are never arguments.

    Predeclared descriptive flag: final development median outside the pooled
    cross-fitted development p10/p90 interval. A flag is evidence about the direct
    transfer assumption, not a statistical guarantee or a new selection rule.
    """
    def population(cross, final):
        a, b = score_distribution(cross), score_distribution(final)
        return TransferPopulationDiagnostics(a, b, b.mean - a.mean, b.median - a.median,
                    float((final >= threshold).mean() - (cross >= threshold).mean()), not a.p10 <= b.median <= a.p90)
    known, unknown = population(cross_known, final_known), population(cross_unknown, final_unknown)
    return ThresholdTransport(known, unknown, known.median_outside_cross_fitted_p10_p90 or unknown.median_outside_cross_fitted_p10_p90)
