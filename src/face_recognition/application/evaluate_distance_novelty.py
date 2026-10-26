"""Known-fit distance references, fold-local calibration, no holdout payloads."""

from dataclasses import replace
from statistics import fmean, pstdev
from typing import Protocol

import numpy as np

from face_recognition.application.calibrate_unknown_rejection import summarize_metrics, two_fold_midpoint, validate_rejection_data
from face_recognition.application.rejection_data import UnknownCalibrationData
from face_recognition.application.rejection_metrics import evaluate_threshold
from face_recognition.application.rejection_score_analysis import analyze_scores, score_distribution
from face_recognition.application.training_folds import DevelopmentDataset, create_session_folds
from face_recognition.application.unknown_rejection_split import UnknownRejectionError
from face_recognition.domain.models.anova_svm_tuning import AnovaSvmCandidate
from face_recognition.domain.models.baseline_evaluation import BaselineModelConfiguration, ClassificationMetrics, EvaluationVersions
from face_recognition.domain.models.distance_novelty import (
    NOVELTY_MODEL, NOVELTY_NEIGHBORS, REFERENCE_SCORES, DistanceFoldResult, DistanceNoveltyReport,
    DistanceNoveltyResult, NeighborIdentityDiagnostics, NoveltyReferenceComparison,
)


class ScoreObservations(Protocol):
    family: str
    known: np.ndarray
    unknown: np.ndarray


class DistanceFoldObservations(Protocol):
    scores: tuple[ScoreObservations, ...]
    known_correct: np.ndarray
    closed_set_metrics: ClassificationMetrics
    nearest_same_identity_rate: float
    unknown_nearest_majority_concentration: float


class DistanceNoveltyRunner(Protocol):
    versions: EvaluationVersions

    def configuration_for(self, candidate: AnovaSvmCandidate) -> BaselineModelConfiguration: ...

    def score_novelty_fold(self, X_fit: np.ndarray, y_fit: np.ndarray, X_validation: np.ndarray,
                           y_validation: np.ndarray, X_unknown: np.ndarray, identity_count: int) -> DistanceFoldObservations: ...


def rank_novelty_results(results: tuple[DistanceNoveltyResult, ...]) -> tuple[str, ...]:
    rows = sorted(results, key=lambda row: (-row.mean_roc_auc, row.mean_eer,
                                           row.mean_metrics.unknown_false_acceptance_rate,
                                           -row.mean_metrics.known_correct_identification_rate,
                                           -row.mean_metrics.open_set_exact_accuracy,
                                           row.mean_metrics.known_rejection_rate, row.neighbor_count))
    return tuple(row.score_family for row in rows)


class DistanceNoveltyEvaluator:
    def __init__(self, runner: DistanceNoveltyRunner) -> None:
        self.runner = runner

    def evaluate(self, known: DevelopmentDataset, unknown: UnknownCalibrationData) -> DistanceNoveltyReport:
        context = "Invalid authorized novelty data"
        try:
            validate_rejection_data(known, unknown, minimum_features=NOVELTY_MODEL.k)
            folds = create_session_folds(known)
            if any(len(fold.fit.indices) < 5 for fold in folds):
                raise UnknownRejectionError("At least 5 known fit samples per fold are required for 1/3/5-neighbor novelty")
            expected = tuple(name for name, _ in NOVELTY_NEIGHBORS) + REFERENCE_SCORES
            observations, analyses, diagnostics = [], [], []
            for fold in folds:
                context = f"Distance novelty evaluation failed: {fold.name}"
                output = self.runner.score_novelty_fold(*fold.fit.arrays(), *fold.validation.arrays(), unknown.X,
                                                        len(known.label_to_person))
                if (tuple(row.family for row in output.scores) != expected
                        or output.known_correct.shape != (len(fold.validation.indices),)
                        or output.known_correct.dtype != np.bool_
                        or not all(np.isfinite(value) and 0 <= value <= 1
                                   for value in (output.nearest_same_identity_rate, output.unknown_nearest_majority_concentration))
                        or any(row.known.shape != (len(fold.validation.indices),) or row.unknown.shape != (len(unknown.samples),)
                               for row in output.scores)
                        or any(np.any(row.known > 0) or np.any(row.unknown > 0) for row in output.scores[:3])):
                    raise ValueError
                analyses.append(tuple(analyze_scores(row.known, output.known_correct, row.unknown) for row in output.scores))
                observations.append(output)
                diagnostics.append(NeighborIdentityDiagnostics(
                    fold.name, len(fold.fit.indices), len(fold.validation.indices), len(unknown.samples),
                    output.nearest_same_identity_rate, output.unknown_nearest_majority_concentration,
                ))
            results = []
            for index, family in enumerate(expected):
                context = f"Distance novelty aggregation failed: {family}"
                neighbor_count = dict(NOVELTY_NEIGHBORS).get(family)
                per_fold = [row[index] for row in analyses]
                thresholds = [row.calibration.threshold for row in per_fold]
                median = two_fold_midpoint(*thresholds, allow_negative=True)
                rows = tuple(DistanceFoldResult(
                    fold.name, fold.fit_session_ids, fold.validation_session_ids, len(fold.fit.indices),
                    len(fold.validation.indices), len(unknown.samples), output.closed_set_metrics, analysis,
                    evaluate_threshold(output.scores[index].known, output.known_correct, output.scores[index].unknown,
                                       median, allow_negative=True),
                    score_distribution(-output.scores[index].known) if neighbor_count else None,
                    score_distribution(-output.scores[index].unknown) if neighbor_count else None,
                ) for fold, output, analysis in zip(folds, observations, per_fold))
                auc, eer = [r.roc_auc for r in per_fold], [r.equal_error_point.eer for r in per_fold]
                metrics = [row.analysis.calibration.metrics for row in rows]
                results.append(DistanceNoveltyResult(
                    family, neighbor_count, rows, fmean(auc), pstdev(auc), fmean(eer), pstdev(eer),
                    summarize_metrics(metrics, fmean), summarize_metrics(metrics, pstdev),
                    summarize_metrics([row.closed_set_metrics for row in rows], fmean), median,
                    abs(thresholds[0] - thresholds[1]), summarize_metrics([row.at_development_threshold for row in rows], fmean),
                    score_distribution(np.concatenate([-row.scores[index].known for row in observations])) if neighbor_count else None,
                    score_distribution(np.concatenate([-row.scores[index].unknown for row in observations])) if neighbor_count else None,
                ))
            references = tuple(results[3:])
            novelty = tuple(replace(row, vs_references=tuple(NoveltyReferenceComparison(
                ref.score_family, row.mean_roc_auc - ref.mean_roc_auc, row.mean_eer - ref.mean_eer,
                row.mean_metrics.unknown_false_acceptance_rate - ref.mean_metrics.unknown_false_acceptance_rate,
                row.mean_metrics.known_rejection_rate - ref.mean_metrics.known_rejection_rate,
                row.mean_metrics.known_correct_identification_rate - ref.mean_metrics.known_correct_identification_rate,
                row.mean_metrics.open_set_exact_accuracy - ref.mean_metrics.open_set_exact_accuracy,
            ) for ref in references)) for row in results[:3])
            ranking = rank_novelty_results(novelty)
            return DistanceNoveltyReport(
                known.X.shape[1], len(known.label_to_person), known.label_to_person, NOVELTY_MODEL,
                self.runner.configuration_for(NOVELTY_MODEL), unknown.partition, novelty, references, tuple(diagnostics),
                fmean(row.nearest_same_identity_rate for row in diagnostics),
                fmean(row.unknown_nearest_majority_concentration for row in diagnostics),
                ranking, ranking[0], self.runner.versions,
            )
        except UnknownRejectionError:
            raise
        except Exception:
            raise UnknownRejectionError(context) from None
