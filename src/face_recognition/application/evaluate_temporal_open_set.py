"""Offline non-overlapping temporal evaluation with fold-local thresholds."""

from dataclasses import dataclass, field, replace
from statistics import fmean, pstdev
from typing import Protocol

import numpy as np

from face_recognition.application.calibrate_unknown_rejection import summarize_metrics, two_fold_midpoint, validate_rejection_data
from face_recognition.application.rejection_data import UnknownCalibrationData
from face_recognition.application.rejection_metrics import evaluate_threshold
from face_recognition.application.rejection_score_analysis import analyze_scores
from face_recognition.application.temporal_windows import WindowObservations, aggregate_windows, build_windows
from face_recognition.application.training_folds import DevelopmentDataset, create_session_folds
from face_recognition.application.unknown_rejection_split import UnknownRejectionError
from face_recognition.domain.models.anova_svm_tuning import AnovaSvmCandidate
from face_recognition.domain.models.baseline_evaluation import BaselineModelConfiguration, ClassificationMetrics, EvaluationVersions
from face_recognition.domain.models.dataset import CONDITIONS
from face_recognition.domain.models.rejection_score_evaluation import ScoreAnalysis
from face_recognition.domain.models.temporal_open_set import (
    TEMPORAL_MODEL, TEMPORAL_SCORES, WINDOW_SIZES, ConditionTemporalDiagnostics, FrameReferenceComparison,
    TemporalFoldResult, TemporalOpenSetReport, TemporalResult, WindowCounts,
)


class FrameInference(Protocol):
    predicted: np.ndarray
    decisions: np.ndarray
    confidence: tuple[tuple[str, np.ndarray], ...]


class TemporalFoldInference(Protocol):
    known: FrameInference
    unknown: FrameInference
    closed_set_metrics: ClassificationMetrics


class TemporalRunner(Protocol):
    versions: EvaluationVersions

    def configuration_for(self, candidate: AnovaSvmCandidate) -> BaselineModelConfiguration: ...

    def infer_fold(self, X_fit: np.ndarray, y_fit: np.ndarray, X_validation: np.ndarray,
                   y_validation: np.ndarray, X_unknown: np.ndarray, identity_count: int) -> TemporalFoldInference: ...


@dataclass(frozen=True, eq=False)
class _WindowCase:
    known: WindowObservations = field(repr=False)
    unknown: WindowObservations = field(repr=False)
    correct: np.ndarray = field(repr=False)
    known_counts: WindowCounts
    unknown_counts: WindowCounts
    known_conditions: tuple[tuple[str, WindowCounts], ...]
    unknown_conditions: tuple[tuple[str, WindowCounts], ...]
    analyses: tuple[ScoreAnalysis, ...]
    frame_metrics: ClassificationMetrics


def _sum_counts(counts) -> WindowCounts:
    return WindowCounts(*(sum(getattr(row, name) for row in counts) for name in WindowCounts.__dataclass_fields__))


def _condition_diagnostics(case: _WindowCase, score_index: int, threshold: float) -> tuple[ConditionTemporalDiagnostics, ...]:
    known_counts, unknown_counts = dict(case.known_conditions), dict(case.unknown_conditions)
    known_conditions, unknown_conditions = np.asarray(case.known.conditions), np.asarray(case.unknown.conditions)
    known_scores, unknown_scores = case.known.confidence[score_index][1], case.unknown.confidence[score_index][1]
    empty = WindowCounts(0, 0, 0, 0)
    rows = []
    for condition in CONDITIONS:
        if condition not in known_counts and condition not in unknown_counts:
            continue
        kmask, umask = known_conditions == condition, unknown_conditions == condition
        nk, nu = int(kmask.sum()), int(umask.sum())
        rows.append(ConditionTemporalDiagnostics(
            condition, known_counts.get(condition, empty), unknown_counts.get(condition, empty),
            float(((known_scores[kmask] >= threshold) & case.correct[kmask]).sum()) / nk if nk else None,
            float((known_scores[kmask] < threshold).sum()) / nk if nk else None,
            float((unknown_scores[umask] < threshold).sum()) / nu if nu else None,
        ))
    return tuple(rows)


def _aggregate_conditions(folds: tuple[TemporalFoldResult, ...]) -> tuple[ConditionTemporalDiagnostics, ...]:
    rows = []
    for condition in CONDITIONS:
        available = [row for fold in folds for row in fold.conditions if row.condition == condition]
        if not available:
            continue

        def rate(name):
            values = [getattr(row, name) for row in available if getattr(row, name) is not None]
            return fmean(values) if values else None

        rows.append(ConditionTemporalDiagnostics(
            condition, _sum_counts([row.known_counts for row in available]), _sum_counts([row.unknown_counts for row in available]),
            rate("known_correct_identification_rate"), rate("known_rejection_rate"), rate("unknown_rejection_rate"),
        ))
    return tuple(rows)


def rank_temporal_results(results: tuple[TemporalResult, ...]) -> tuple[str, ...]:
    rows = sorted(results, key=lambda row: (-row.mean_roc_auc, row.mean_eer,
                                           row.mean_metrics.unknown_false_acceptance_rate,
                                           -row.mean_metrics.known_correct_identification_rate,
                                           -row.mean_metrics.open_set_exact_accuracy,
                                           row.mean_metrics.known_rejection_rate, row.window_size, row.score_family))
    return tuple(row.identifier for row in rows)


class TemporalOpenSetEvaluator:
    def __init__(self, runner: TemporalRunner) -> None:
        self.runner = runner

    def evaluate(self, known: DevelopmentDataset, unknown: UnknownCalibrationData) -> TemporalOpenSetReport:
        try:
            completed = self.evaluate_results(known, unknown)
            ranking = rank_temporal_results(completed)
            return TemporalOpenSetReport(known.X.shape[1], len(known.label_to_person), known.label_to_person, TEMPORAL_MODEL,
                                          self.runner.configuration_for(TEMPORAL_MODEL), unknown.partition,
                                          completed, ranking, ranking[0], self.runner.versions)
        except UnknownRejectionError:
            raise
        except Exception:
            raise UnknownRejectionError("Temporal report construction failed") from None

    def evaluate_results(self, known: DevelopmentDataset, unknown: UnknownCalibrationData, *,
                         score_families: tuple[str, ...] = TEMPORAL_SCORES,
                         minimum_fit_samples: int = 1) -> tuple[TemporalResult, ...]:
        """Shared fixed-model window/calibration core; callers supply score outputs.

        No classifier or score is fitted here. Existing and distance-novelty
        runners provide fold-local predictions and confidence arrays. Reports
        and ranking remain specific to each experiment.
        """
        context = "Invalid authorized temporal data"
        try:
            if (not score_families or len(set(score_families)) != len(score_families)
                    or any(not isinstance(name, str) or not name for name in score_families)
                    or type(minimum_fit_samples) is not int or minimum_fit_samples < 1):
                raise ValueError
            validate_rejection_data(known, unknown, minimum_features=TEMPORAL_MODEL.k)
            folds = create_session_folds(known)
            if any(len(fold.fit.indices) < minimum_fit_samples for fold in folds):
                raise UnknownRejectionError(f"At least {minimum_fit_samples} known fit samples per fold are required")
            labels = {person: index for index, person in enumerate(known.label_to_person)}
            cases = []
            for fold in folds:
                context = f"Temporal inference failed: {fold.name}"
                output = self.runner.infer_fold(*fold.fit.arrays(), *fold.validation.arrays(), unknown.X, len(labels))
                if (tuple(name for name, _ in output.known.confidence) != score_families
                        or tuple(name for name, _ in output.unknown.confidence) != score_families
                        or output.known.decisions.shape != (len(fold.validation.indices), len(labels))
                        or output.unknown.decisions.shape != (len(unknown.samples), len(labels))):
                    raise ValueError
                by_size = {}
                for size in WINDOW_SIZES:
                    context = f"Temporal evaluation failed: {fold.name}, window={size}"
                    known_plan, unknown_plan = build_windows(fold.validation.metadata, size), build_windows(unknown.samples, size)
                    if not known_plan.windows or not unknown_plan.windows:
                        raise UnknownRejectionError(f"Window {size} has no complete known or unknown calibration windows")
                    kw = aggregate_windows(known_plan, output.known.predicted, output.known.decisions, output.known.confidence)
                    uw = aggregate_windows(unknown_plan, output.unknown.predicted, output.unknown.decisions, output.unknown.confidence)
                    truth = np.fromiter((labels[window.person_id] for window in known_plan.windows), dtype=np.int32)
                    correct = kw.predicted == truth
                    analyses = tuple(analyze_scores(kvalues, correct, uvalues)
                                     for (_, kvalues), (_, uvalues) in zip(kw.confidence, uw.confidence))
                    by_size[size] = _WindowCase(kw, uw, correct, known_plan.counts, unknown_plan.counts,
                                                known_plan.condition_counts, unknown_plan.condition_counts,
                                                analyses, output.closed_set_metrics)
                cases.append(by_size)
                # Per-frame decisions/models do not survive into the next fit.
                del output, known_plan, unknown_plan
            results = []
            for index, family in enumerate(score_families):
                for size in WINDOW_SIZES:
                    selected = [case[size] for case in cases]
                    thresholds = [case.analyses[index].calibration.threshold for case in selected]
                    median = two_fold_midpoint(*thresholds, allow_negative=True)
                    rows = tuple(TemporalFoldResult(
                        fold.name, fold.fit_session_ids, fold.validation_session_ids, len(fold.fit.indices),
                        case.known_counts, case.unknown_counts, case.frame_metrics, case.analyses[index],
                        evaluate_threshold(case.known.confidence[index][1], case.correct, case.unknown.confidence[index][1],
                                           median, allow_negative=True),
                        _condition_diagnostics(case, index, case.analyses[index].calibration.threshold), case.known.vote_stability,
                    ) for fold, case in zip(folds, selected))
                    auc, eer = [r.analysis.roc_auc for r in rows], [r.analysis.equal_error_point.eer for r in rows]
                    metrics = [r.analysis.calibration.metrics for r in rows]
                    results.append(TemporalResult(
                        family, size, rows, fmean(auc), pstdev(auc), fmean(eer), pstdev(eer),
                        summarize_metrics(metrics, fmean), summarize_metrics(metrics, pstdev), median,
                        abs(thresholds[0] - thresholds[1]), summarize_metrics([r.at_development_threshold for r in rows], fmean),
                        _sum_counts([r.known_counts for r in rows]), _sum_counts([r.unknown_counts for r in rows]),
                        summarize_metrics([r.known_vote_stability.rates for r in rows], fmean), _aggregate_conditions(rows),
                    ))
            references = {row.score_family: row for row in results if row.window_size == 1}
            completed = tuple(replace(row, vs_window1=FrameReferenceComparison(
                row.mean_roc_auc - references[row.score_family].mean_roc_auc,
                row.mean_eer - references[row.score_family].mean_eer,
                row.mean_metrics.unknown_false_acceptance_rate - references[row.score_family].mean_metrics.unknown_false_acceptance_rate,
                row.mean_metrics.known_rejection_rate - references[row.score_family].mean_metrics.known_rejection_rate,
                row.mean_metrics.known_correct_identification_rate - references[row.score_family].mean_metrics.known_correct_identification_rate,
                row.mean_metrics.open_set_exact_accuracy - references[row.score_family].mean_metrics.open_set_exact_accuracy,
            )) for row in results)
            return completed
        except UnknownRejectionError:
            raise
        except Exception:
            raise UnknownRejectionError(context) from None
