"""Controlled development search; protected partitions never supply features."""

from dataclasses import replace
from typing import Protocol

import numpy as np

from face_recognition.application.evaluate_baseline_models import BaselineIdentityEvaluator, BaselineRunner
from face_recognition.application.training_folds import DevelopmentDataset
from face_recognition.domain.models.anova_svm_tuning import (
    ANOVA_K_VALUES, REFERENCE_CANDIDATE, SVM_GAMMA_VALUES, AnovaSvmCandidate, AnovaSvmResult,
    AnovaSvmTuningReport, FoldSupportVectors, RankedTuningCandidate, SelectionStability,
    TuningFoldMeasurements, tuning_candidates,
)
from face_recognition.domain.models.baseline_evaluation import BaselineClassifierResult, BaselineModelConfiguration, FoldMeasurements
from face_recognition.domain.models.feature_reduction import BaselineComparison


class AnovaSvmTuningError(ValueError):
    pass


class AnovaSvmRunner(BaselineRunner, Protocol):
    def configuration_for(self, candidate: AnovaSvmCandidate) -> BaselineModelConfiguration: ...

    def run_candidate(self, candidate: AnovaSvmCandidate, X_fit: np.ndarray, y_fit: np.ndarray,
                      X_validation: np.ndarray, y_validation: np.ndarray, identity_count: int) -> TuningFoldMeasurements: ...


class _CandidateAdapter:
    def __init__(self, runner: AnovaSvmRunner, candidate: AnovaSvmCandidate) -> None:
        self.runner, self.candidate = runner, candidate
        self.configurations = (runner.configuration_for(candidate),)
        self.versions = runner.versions
        self.diagnostics: list[TuningFoldMeasurements] = []

    def run(self, configuration: BaselineModelConfiguration, X_fit: np.ndarray, y_fit: np.ndarray,
            X_validation: np.ndarray, y_validation: np.ndarray, identity_count: int) -> FoldMeasurements:
        result = self.runner.run_candidate(self.candidate, X_fit, y_fit, X_validation, y_validation, identity_count)
        self.diagnostics.append(result)
        return result.measurements


def selection_stability(k: int, first: tuple[int, ...], second: tuple[int, ...],
                        fold_names: tuple[str, str] = ("Fold A", "Fold B")) -> SelectionStability:
    if type(k) is not int or k < 1 or any(
        len(indices) != k or tuple(sorted(set(indices))) != indices
        or any(type(index) is not int or index < 0 for index in indices) for indices in (first, second)
    ):
        raise AnovaSvmTuningError("Invalid selected-feature index metadata")
    shared, union = len(set(first) & set(second)), len(set(first) | set(second))
    return SelectionStability(k, *fold_names, first, second, shared, 100 * shared / k, shared / union)


def rank_tuning_results(results: tuple[AnovaSvmResult, ...]) -> tuple[RankedTuningCandidate, ...]:
    ordered = sorted(results, key=lambda row: (
        -row.evaluation.mean_metrics.macro_f1, -row.worst_fold_macro_f1, row.fold_f1_gap,
        -row.evaluation.mean_metrics.accuracy, row.candidate.k,
        row.evaluation.mean_timing.mean_prediction_latency_ms, row.candidate.C,
        SVM_GAMMA_VALUES.index(row.candidate.gamma),
    ))
    return tuple(RankedTuningCandidate(index, row.candidate.identifier) for index, row in enumerate(ordered, 1))


def _comparison(candidate: AnovaSvmResult, reference: BaselineClassifierResult, reference_features: int) -> BaselineComparison:
    dimension_reduction = reference_features - candidate.candidate.k
    return BaselineComparison(candidate.evaluation.mean_metrics.macro_f1 - reference.mean_metrics.macro_f1,
                              candidate.evaluation.mean_metrics.accuracy - reference.mean_metrics.accuracy,
                              dimension_reduction, 100 * dimension_reduction / reference_features,
                              candidate.evaluation.mean_timing.mean_prediction_latency_ms
                              - reference.mean_timing.mean_prediction_latency_ms)


class AnovaSvmTuner:
    def __init__(self, runner: AnovaSvmRunner) -> None:
        self.runner = runner

    def evaluate(self, data: DevelopmentDataset) -> AnovaSvmTuningReport:
        context = "Invalid development data or fixed tuning grid"
        try:
            if data.X.shape[1] < max(ANOVA_K_VALUES):
                raise AnovaSvmTuningError("The fixed tuning grid requires at least 768 source features; nothing was fitted")
            # This also revalidates finite features, labels, roles and complete
            # session coverage, without copying source arrays or scanning pools.
            baseline_report = BaselineIdentityEvaluator(self.runner).evaluate(data)
            if len(baseline_report.classifiers) != 1 or baseline_report.classifiers[0].configuration.name != "SVM":
                raise ValueError
            baseline = baseline_report.classifiers[0]
            results, stability = [], {}
            for candidate in tuning_candidates():
                context = f"ANOVA-SVM tuning failed: {candidate.identifier}"
                adapter = _CandidateAdapter(self.runner, candidate)
                evaluation = BaselineIdentityEvaluator(adapter).evaluate(data).classifiers[0]
                if len(evaluation.folds) != 2 or len(adapter.diagnostics) != 2:
                    raise ValueError
                names = tuple(fold.fold_name for fold in evaluation.folds)
                indices = tuple(row.selected_feature_indices for row in adapter.diagnostics)
                if any(index >= data.X.shape[1] for selected in indices for index in selected):
                    raise ValueError
                selected = selection_stability(candidate.k, *indices, names)
                if stability.setdefault(candidate.k, selected) != selected:
                    raise AnovaSvmTuningError("ANOVA selections changed across C/gamma for the same k and fold")
                for fold, diagnostics in zip(evaluation.folds, adapter.diagnostics):
                    counts = diagnostics.support_vectors
                    if len(counts.per_class) != len(data.label_to_person) or counts.total > fold.fit_samples:
                        raise ValueError
                evaluation = replace(evaluation, folds=tuple(replace(fold, feature_count=candidate.k) for fold in evaluation.folds))
                f1 = tuple(fold.measurements.metrics.macro_f1 for fold in evaluation.folds)
                results.append(AnovaSvmResult(candidate, evaluation,
                                              tuple(FoldSupportVectors(fold.fold_name, row.support_vectors)
                                                    for fold, row in zip(evaluation.folds, adapter.diagnostics)),
                                              min(f1), abs(f1[0] - f1[1])))
                # Only small counts/metrics and one index pair per k survive.
                # The runner has already released its fitted pipeline/matrices.
            reference = next(row for row in results if row.candidate == REFERENCE_CANDIDATE)
            completed = tuple(replace(row, vs_full_svm_baseline=_comparison(row, baseline, data.X.shape[1]),
                                      vs_anova512_reference=_comparison(row, reference.evaluation, REFERENCE_CANDIDATE.k))
                              for row in results)
            ranking = rank_tuning_results(completed)
            selected_candidate = next(row.candidate for row in completed if row.candidate.identifier == ranking[0].candidate_id)
            return AnovaSvmTuningReport(data.X.shape[1], len(data.label_to_person), len(data.samples), data.label_to_person,
                                        baseline, completed, tuple(stability[k] for k in ANOVA_K_VALUES), ranking,
                                        selected_candidate, self.runner.versions)
        except AnovaSvmTuningError:
            raise
        except Exception:
            raise AnovaSvmTuningError(context) from None
