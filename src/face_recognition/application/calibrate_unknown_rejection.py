"""Four known-only fits, fold-local calibration, development-only comparisons."""

from math import isfinite
from statistics import fmean, pstdev
from typing import Protocol

import numpy as np

from face_recognition.application.rejection_data import UnknownCalibrationData
from face_recognition.application.rejection_metrics import calibrate_threshold, evaluate_threshold
from face_recognition.application.training_folds import DevelopmentDataset, create_session_folds
from face_recognition.application.unknown_rejection_split import UnknownRejectionError
from face_recognition.domain.models.anova_svm_tuning import AnovaSvmCandidate
from face_recognition.domain.models.baseline_evaluation import BaselineModelConfiguration, ClassificationMetrics, EvaluationVersions
from face_recognition.domain.models.identity_split import IdentityPartition
from face_recognition.domain.models.unknown_rejection import (
    REJECTION_CANDIDATES, RejectionCandidateResult, RejectionFoldResult, UnknownRejectionReport,
)


class RejectionScoreResult(Protocol):
    known_confidence: np.ndarray
    known_correct: np.ndarray
    unknown_confidence: np.ndarray
    closed_set_metrics: ClassificationMetrics


class RejectionRunner(Protocol):
    versions: EvaluationVersions

    def configuration_for(self, candidate: AnovaSvmCandidate) -> BaselineModelConfiguration: ...

    def score_fold(self, candidate: AnovaSvmCandidate, X_fit: np.ndarray, y_fit: np.ndarray, X_validation: np.ndarray,
                   y_validation: np.ndarray, X_unknown: np.ndarray, identity_count: int) -> RejectionScoreResult: ...


def two_fold_midpoint(first: float, second: float, *, allow_negative: bool = False) -> float:
    if not all(isfinite(value) and (allow_negative or value >= 0) for value in (first, second)):
        raise UnknownRejectionError("Invalid fold thresholds")
    low, high = sorted((first, second))
    return low + (high - low) / 2


def rank_rejection_candidates(results: tuple[RejectionCandidateResult, ...]) -> tuple[str, ...]:
    ordered = sorted(results, key=lambda row: (-row.mean_metrics.open_set_exact_accuracy,
                                               row.mean_metrics.unknown_false_acceptance_rate,
                                               -row.mean_metrics.known_correct_identification_rate,
                                               -row.mean_metrics.binary_balanced_accuracy,
                                               row.absolute_threshold_difference, row.candidate.k,
                                               row.candidate.identifier))
    return tuple(row.candidate.identifier for row in ordered)


def summarize_metrics(values, statistic):
    model = type(values[0])
    return model(**{name: statistic(getattr(row, name) for row in values) for name in model.__dataclass_fields__})


def validate_rejection_data(known: DevelopmentDataset, unknown: UnknownCalibrationData) -> None:
    DevelopmentDataset(known.X, known.y, known.samples, known.label_to_person, known.training_sessions)
    UnknownCalibrationData(unknown.X, unknown.samples, unknown.source, unknown.partition)
    expected_known = {row for row in unknown.source.samples if row.partition is IdentityPartition.KNOWN_TRAIN_POOL}
    if (set(known.samples) != expected_known or known.label_to_person != unknown.source.specification.known_ids
            or known.training_sessions != unknown.source.specification.train_sessions
            or known.X.shape[1] != unknown.X.shape[1] or known.X.shape[1] < max(c.k for c in REJECTION_CANDIDATES)
            or len(known.label_to_person) < 2):
        raise UnknownRejectionError("Invalid authorized rejection calibration data")


class UnknownRejectionCalibrator:
    def __init__(self, runner: RejectionRunner) -> None:
        self.runner = runner

    def evaluate(self, known: DevelopmentDataset, unknown: UnknownCalibrationData) -> UnknownRejectionReport:
        context = "Invalid authorized rejection calibration data"
        try:
            validate_rejection_data(known, unknown)
            folds = create_session_folds(known)
            results = []
            for candidate in REJECTION_CANDIDATES:
                scored, calibrations = [], []
                for fold in folds:
                    context = f"Rejection calibration failed: {candidate.identifier}, {fold.name}"
                    scores = self.runner.score_fold(candidate, *fold.fit.arrays(), *fold.validation.arrays(),
                                                    unknown.X, len(known.label_to_person))
                    if (len(scores.known_confidence) != len(fold.validation.indices)
                            or len(scores.unknown_confidence) != len(unknown.samples)):
                        raise ValueError
                    calibrations.append(calibrate_threshold(scores.known_confidence, scores.known_correct, scores.unknown_confidence))
                    scored.append(scores)
                midpoint = two_fold_midpoint(*(row.threshold for row in calibrations))
                rows = tuple(RejectionFoldResult(
                    fold.name, fold.fit_session_ids, fold.validation_session_ids, len(fold.fit.indices),
                    len(fold.validation.indices), len(unknown.samples), scores.closed_set_metrics, calibration,
                    evaluate_threshold(scores.known_confidence, scores.known_correct, scores.unknown_confidence, midpoint),
                ) for fold, scores, calibration in zip(folds, scored, calibrations))
                results.append(RejectionCandidateResult(
                    candidate, self.runner.configuration_for(candidate), rows,
                    summarize_metrics([row.calibration.metrics for row in rows], fmean),
                    summarize_metrics([row.calibration.metrics for row in rows], pstdev),
                    summarize_metrics([row.at_development_threshold for row in rows], fmean),
                    summarize_metrics([row.closed_set_metrics for row in rows], fmean),
                    summarize_metrics([row.closed_set_metrics for row in rows], pstdev),
                    midpoint, midpoint, abs(calibrations[0].threshold - calibrations[1].threshold), midpoint,
                ))
                # Keep only aggregate results. Reuse the source matrices for the
                # next candidate and release this candidate's confidence arrays.
                del scored, calibrations, scores
            ranking = rank_rejection_candidates(tuple(results))
            selected = next(row.candidate for row in results if row.candidate.identifier == ranking[0])
            return UnknownRejectionReport(known.X.shape[1], len(known.label_to_person), known.label_to_person,
                                           unknown.partition, tuple(results), ranking, selected, self.runner.versions)
        except UnknownRejectionError:
            raise
        except Exception:
            raise UnknownRejectionError(context) from None
