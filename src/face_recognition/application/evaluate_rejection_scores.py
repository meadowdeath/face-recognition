"""Four fixed score families, two fixed models, no protected payload access."""

from dataclasses import replace
from statistics import fmean, pstdev
from typing import Protocol

import numpy as np

from face_recognition.application.calibrate_unknown_rejection import (
    summarize_metrics, two_fold_midpoint, validate_rejection_data,
)
from face_recognition.application.rejection_data import UnknownCalibrationData
from face_recognition.application.rejection_metrics import evaluate_threshold
from face_recognition.application.rejection_score_analysis import analyze_scores, score_distribution
from face_recognition.application.training_folds import DevelopmentDataset, create_session_folds
from face_recognition.application.unknown_rejection_split import UnknownRejectionError
from face_recognition.domain.models.anova_svm_tuning import AnovaSvmCandidate
from face_recognition.domain.models.baseline_evaluation import BaselineModelConfiguration, ClassificationMetrics, EvaluationVersions
from face_recognition.domain.models.rejection_score_evaluation import (
    SCORE_FAMILIES, MarginReferenceComparison, RejectionScoreReport, ScoreEvaluationResult, ScoreFoldGaps, ScoreFoldResult,
)
from face_recognition.domain.models.unknown_rejection import REJECTION_CANDIDATES


class ScoreObservations(Protocol):
    family: str
    known: np.ndarray
    unknown: np.ndarray


class FoldScoreObservations(Protocol):
    scores: tuple[ScoreObservations, ...]
    known_correct: np.ndarray
    closed_set_metrics: ClassificationMetrics


class ScoreRunner(Protocol):
    versions: EvaluationVersions

    def configuration_for(self, candidate: AnovaSvmCandidate) -> BaselineModelConfiguration: ...

    def score_fold_all(self, candidate: AnovaSvmCandidate, X_fit: np.ndarray, y_fit: np.ndarray,
                       X_validation: np.ndarray, y_validation: np.ndarray,
                       X_unknown: np.ndarray, identity_count: int) -> FoldScoreObservations: ...


def rank_score_results(results: tuple[ScoreEvaluationResult, ...]) -> tuple[str, ...]:
    rows = sorted(results, key=lambda row: (-row.mean_roc_auc, row.mean_eer,
                                           row.mean_metrics.unknown_false_acceptance_rate,
                                           -row.mean_metrics.known_correct_identification_rate,
                                           -row.mean_metrics.open_set_exact_accuracy,
                                           row.absolute_threshold_difference, row.candidate.k, row.score_family))
    return tuple(row.identifier for row in rows)


class RejectionScoreEvaluator:
    def __init__(self, runner: ScoreRunner) -> None:
        self.runner = runner

    def evaluate(self, known: DevelopmentDataset, unknown: UnknownCalibrationData) -> RejectionScoreReport:
        context = "Invalid authorized rejection-score data"
        try:
            validate_rejection_data(known, unknown)
            folds = create_session_folds(known)
            results = []
            for candidate in REJECTION_CANDIDATES:
                observations, analyses = [], []
                for fold in folds:
                    context = f"Rejection score evaluation failed: {candidate.identifier}, {fold.name}"
                    output = self.runner.score_fold_all(candidate, *fold.fit.arrays(), *fold.validation.arrays(),
                                                         unknown.X, len(known.label_to_person))
                    if (tuple(row.family for row in output.scores) != SCORE_FAMILIES
                            or len(output.known_correct) != len(fold.validation.indices)
                            or any(len(row.known) != len(fold.validation.indices) or len(row.unknown) != len(unknown.samples)
                                   for row in output.scores)):
                        raise ValueError
                    analyses.append(tuple(analyze_scores(row.known, output.known_correct, row.unknown) for row in output.scores))
                    observations.append(output)
                for index, family in enumerate(SCORE_FAMILIES):
                    per_fold = [row[index] for row in analyses]
                    thresholds = [row.calibration.threshold for row in per_fold]
                    median = two_fold_midpoint(*thresholds, allow_negative=True)
                    rows = tuple(ScoreFoldResult(
                        fold.name, fold.fit_session_ids, fold.validation_session_ids, len(fold.fit.indices),
                        len(fold.validation.indices), len(unknown.samples), output.closed_set_metrics, analysis,
                        evaluate_threshold(output.scores[index].known, output.known_correct, output.scores[index].unknown,
                                           median, allow_negative=True),
                    ) for fold, output, analysis in zip(folds, observations, per_fold))
                    metrics = [row.analysis.calibration.metrics for row in rows]
                    auc, eer = [row.roc_auc for row in per_fold], [row.equal_error_point.eer for row in per_fold]
                    results.append(ScoreEvaluationResult(
                        candidate, family, self.runner.configuration_for(candidate), rows,
                        fmean(auc), pstdev(auc), fmean(eer), pstdev(eer),
                        summarize_metrics(metrics, fmean), summarize_metrics(metrics, pstdev),
                        summarize_metrics([row.closed_set_metrics for row in rows], fmean),
                        median, abs(thresholds[0] - thresholds[1]),
                        summarize_metrics([row.at_development_threshold for row in rows], fmean),
                        ScoreFoldGaps(abs(auc[0] - auc[1]),
                                      abs(metrics[0].unknown_false_acceptance_rate - metrics[1].unknown_false_acceptance_rate),
                                      abs(metrics[0].known_rejection_rate - metrics[1].known_rejection_rate),
                                      abs(metrics[0].open_set_exact_accuracy - metrics[1].open_set_exact_accuracy)),
                        score_distribution(np.concatenate([row.scores[index].known for row in observations])),
                        score_distribution(np.concatenate([row.scores[index].unknown for row in observations])),
                    ))
                del observations, analyses, output
            margin = {row.candidate: row for row in results if row.score_family == "svm_margin"}
            completed = tuple(replace(row, vs_margin=MarginReferenceComparison(
                row.mean_roc_auc - margin[row.candidate].mean_roc_auc,
                row.mean_eer - margin[row.candidate].mean_eer,
                row.mean_metrics.unknown_false_acceptance_rate - margin[row.candidate].mean_metrics.unknown_false_acceptance_rate,
                row.mean_metrics.known_rejection_rate - margin[row.candidate].mean_metrics.known_rejection_rate,
                row.mean_metrics.known_correct_identification_rate - margin[row.candidate].mean_metrics.known_correct_identification_rate,
                row.mean_metrics.open_set_exact_accuracy - margin[row.candidate].mean_metrics.open_set_exact_accuracy,
            )) for row in results)
            ranking = rank_score_results(completed)
            return RejectionScoreReport(known.X.shape[1], len(known.label_to_person), known.label_to_person,
                                         unknown.partition, completed, ranking, ranking[0], self.runner.versions)
        except UnknownRejectionError:
            raise
        except Exception:
            raise UnknownRejectionError(context) from None
