"""Pure synthetic aggregate reports: no feature matrices, files or fitting."""

from dataclasses import replace
import json
from math import inf, nextafter

from face_recognition.domain.models.baseline_evaluation import BaselineModelConfiguration, ClassificationMetrics, EvaluationVersions
from face_recognition.domain.models.distance_novelty import NOVELTY_MODEL
from face_recognition.domain.models.open_set_config import OpenSetArchitecture
from face_recognition.domain.models.rejection_score_evaluation import EqualErrorPoint, FarOperatingPoint, ScoreAnalysis, ScoreDistribution
from face_recognition.domain.models.temporal_distance_novelty import TEMPORAL_DISTANCE_NEIGHBORS, TemporalDistanceNoveltyReport
from face_recognition.domain.models.temporal_open_set import (
    WINDOW_SIZES, ConditionTemporalDiagnostics, FrameReferenceComparison, TemporalFoldResult, TemporalResult,
    VoteRates, VoteStability, WindowCounts,
)
from face_recognition.domain.models.unknown_rejection import (
    RejectionMetrics, ThresholdCalibration, UnknownPartitionCounts, UnknownRejectionSplit, UnknownSplitSpec,
)
from face_recognition.infrastructure.persistence.temporal_distance_novelty_report import temporal_distance_report_to_json


def synthetic_development_report() -> dict:
    rows = []
    zero_metrics = RejectionMetrics(*(0. for _ in range(9)))
    rejected = RejectionMetrics(0., 1., 1., 0., .5, 1 / 3, 0., 0., .5)
    for family, _ in TEMPORAL_DISTANCE_NEIGHBORS:
        for size in WINDOW_SIZES:
            winner = family == "mean_5nn" and size == 7
            metrics = RejectionMetrics(1., 0., 1., 0., 1., 1., 1., 0., 1.) if winner else rejected
            threshold = -1. if winner else nextafter(-1., inf)
            counts = WindowCounts(40, 2 * (20 // size) * size, 40 - 2 * (20 // size) * size, 2 * (20 // size))
            distribution = lambda value: ScoreDistribution(counts.windows, value, 0., value, value, value, value, value)
            analysis = ScoreAnalysis(
                1. if winner else .5, EqualErrorPoint(0. if winner else .5, threshold, 0., 0. if winner else 1.),
                ThresholdCalibration(threshold, metrics, 4 if winner else 2),
                tuple(FarOperatingPoint(target, "available" if winner else "impractical", threshold, metrics,
                                        None if winner else "At most 1% of known samples accepted") for target in (.2, .1, .05)),
                distribution(-1.), distribution(-3. if winner else -1.),
            )
            condition = ConditionTemporalDiagnostics("center", counts, counts, metrics.known_correct_identification_rate,
                                                       metrics.known_rejection_rate, metrics.unknown_rejection_rate)
            folds = tuple(TemporalFoldResult(
                name, (fit,), (validation,), 40, counts, counts, ClassificationMetrics(.8, .8, .8, .8), analysis,
                metrics, (condition,), VoteStability(counts.windows, 0, 0, VoteRates(1., 0., 0.)),
            ) for name, fit, validation in (("Fold A", "visit_a", "visit_b"), ("Fold B", "visit_b", "visit_a")))
            total = WindowCounts(*(2 * value for value in vars(counts).values()))
            rows.append(TemporalResult(
                family, size, folds, analysis.roc_auc, 0., analysis.equal_error_point.eer, 0., metrics, zero_metrics,
                threshold, 0., metrics, total, total, VoteRates(1., 0., 0.),
                (replace(condition, known_counts=total, unknown_counts=total),), FrameReferenceComparison(0., 0., 0., 0., 0., 0.),
            ))
    ranking = tuple(r.identifier for r in sorted(rows, key=lambda r: (-r.mean_roc_auc, r.mean_eer,
                     r.mean_metrics.unknown_false_acceptance_rate, -r.mean_metrics.known_correct_identification_rate,
                     -r.mean_metrics.open_set_exact_accuracy, r.mean_metrics.known_rejection_rate, r.window_size,
                     dict(TEMPORAL_DISTANCE_NEIGHBORS)[r.score_family])))
    architecture = OpenSetArchitecture()
    report = TemporalDistanceNoveltyReport(
        1434, 2, ("synthetic_known_a", "synthetic_known_b"), NOVELTY_MODEL,
        BaselineModelConfiguration("SVM", architecture.svm_parameters, architecture.scaler_parameters),
        UnknownRejectionSplit(UnknownSplitSpec(("synthetic_cal",), ("synthetic_hold",)), "0" * 64, 1, 200,
                              UnknownPartitionCounts(1, 2, 2, 40), UnknownPartitionCounts(1, 2, 2, 40)),
        tuple(rows), ranking, ranking[0], EvaluationVersions("3.11.synthetic", "synthetic", "synthetic", "synthetic"),
    )
    return json.loads(temporal_distance_report_to_json(report))


def synthetic_report_bytes() -> bytes:
    return (json.dumps(synthetic_development_report(), indent=2) + "\n").encode("utf-8")
