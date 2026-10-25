"""Reduction experiment orchestration reusing baseline folds and aggregation."""

from dataclasses import replace
from statistics import fmean, pstdev
from typing import Protocol

import numpy as np

from face_recognition.application.evaluate_baseline_models import BaselineIdentityEvaluator
from face_recognition.application.training_folds import DevelopmentDataset
from face_recognition.domain.models.baseline_evaluation import BaselineModelConfiguration, EvaluationVersions, FoldMeasurements
from face_recognition.domain.models.feature_reduction import (
    DEFAULT_REDUCTIONS, BaselineComparison, FeatureReductionReport, FoldReductionMetadata,
    RankedReduction, ReducedMeasurements, ReductionClassifierResult, ReductionConfiguration,
)


class FeatureReductionError(ValueError):
    pass


class ReductionRunner(Protocol):
    configurations: tuple[BaselineModelConfiguration, ...]
    versions: EvaluationVersions

    def run_reduced(self, reduction: ReductionConfiguration, configuration: BaselineModelConfiguration,
                    X_fit: np.ndarray, y_fit: np.ndarray, X_validation: np.ndarray, y_validation: np.ndarray,
                    identity_count: int) -> ReducedMeasurements: ...


class _BaselineAdapter:
    """Keep only small transformation metadata alongside the baseline run."""

    def __init__(self, runner: ReductionRunner, reduction: ReductionConfiguration) -> None:
        self.runner, self.reduction = runner, reduction
        self.configurations, self.versions = runner.configurations, runner.versions
        self.metadata = []

    def run(self, configuration: BaselineModelConfiguration, X_fit: np.ndarray, y_fit: np.ndarray,
            X_validation: np.ndarray, y_validation: np.ndarray, identity_count: int) -> FoldMeasurements:
        result = self.runner.run_reduced(self.reduction, configuration, X_fit, y_fit, X_validation, y_validation, identity_count)
        self.metadata.append(result.transformation)
        return result.measurements


def rank_results(results: tuple[ReductionClassifierResult, ...]) -> tuple[RankedReduction, ...]:
    ordered = sorted(results, key=lambda row: (-row.evaluation.mean_metrics.macro_f1,
                                               -row.evaluation.mean_metrics.accuracy, row.mean_retained_features,
                                               row.evaluation.mean_timing.mean_prediction_latency_ms,
                                               row.reduction.identifier, row.evaluation.configuration.name))
    return tuple(RankedReduction(index, row.reduction.identifier, row.evaluation.configuration.name)
                 for index, row in enumerate(ordered, 1))


class FeatureReductionEvaluator:
    def __init__(self, runner: ReductionRunner, reductions: tuple[ReductionConfiguration, ...] = DEFAULT_REDUCTIONS) -> None:
        if (not reductions or sum(row.method == "baseline" for row in reductions) != 1
                or len({row.identifier for row in reductions}) != len(reductions)):
            raise FeatureReductionError("Supply one baseline and unique reduction configurations")
        self.runner = runner
        self.reductions = tuple(sorted(reductions, key=lambda row: (
            ("baseline", "anova", "pca").index(row.method), row.k or row.explained_variance or 0,
        )))

    def evaluate(self, data: DevelopmentDataset) -> FeatureReductionReport:
        context = "Invalid development data or feature reduction configuration"
        try:
            if any(row.method == "anova" and row.k > data.X.shape[1] for row in self.reductions):
                raise FeatureReductionError("ANOVA k exceeds source feature count; no candidates were evaluated")
            results, baselines = [], {}
            for reduction in self.reductions:
                context = f"Feature reduction evaluation failed: {reduction.identifier}"
                adapter = _BaselineAdapter(self.runner, reduction)
                report = BaselineIdentityEvaluator(adapter).evaluate(data)
                cursor = 0
                for evaluation in report.classifiers:
                    metadata = tuple(adapter.metadata[cursor:cursor + len(evaluation.folds)])
                    cursor += len(evaluation.folds)
                    if len(metadata) != len(evaluation.folds) or any(
                        type(row.retained_features) is not int or not 1 <= row.retained_features <= data.X.shape[1]
                        for row in metadata
                    ):
                        raise ValueError
                    evaluation = replace(evaluation, folds=tuple(
                        replace(fold, feature_count=transform.retained_features)
                        for fold, transform in zip(evaluation.folds, metadata)
                    ))
                    retained = tuple(row.retained_features for row in metadata)
                    mean_retained = fmean(retained)
                    name = evaluation.configuration.name
                    if reduction.method == "baseline":
                        baselines[name] = evaluation
                    baseline = baselines[name]
                    difference = data.X.shape[1] - mean_retained
                    comparison = BaselineComparison(
                        evaluation.mean_metrics.macro_f1 - baseline.mean_metrics.macro_f1,
                        evaluation.mean_metrics.accuracy - baseline.mean_metrics.accuracy,
                        difference, 100 * difference / data.X.shape[1],
                        evaluation.mean_timing.mean_prediction_latency_ms - baseline.mean_timing.mean_prediction_latency_ms,
                    )
                    results.append(ReductionClassifierResult(
                        reduction, evaluation, tuple(FoldReductionMetadata(fold.fold_name, transform)
                                                    for fold, transform in zip(evaluation.folds, metadata)),
                        mean_retained, pstdev(retained), comparison,
                    ))
                if cursor != len(adapter.metadata):
                    raise ValueError
                # No transformed matrices or fitted models survive this iteration.
            return FeatureReductionReport(data.X.shape[1], len(data.label_to_person), len(data.samples),
                                           data.label_to_person, tuple(results), rank_results(tuple(results)), self.runner.versions)
        except FeatureReductionError:
            raise
        except Exception:
            raise FeatureReductionError(context) from None
