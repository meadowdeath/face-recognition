"""Aggregate development reduction CLI: 0 success, 1 failure, 2 usage error."""

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from face_recognition.application.evaluate_feature_reduction import FeatureReductionError, FeatureReductionEvaluator
from face_recognition.application.validate_identity_split import SplitValidationError
from face_recognition.domain.models.feature_reduction import DEFAULT_REDUCTIONS, ReductionConfiguration
from face_recognition.infrastructure.classifiers.sklearn_feature_reduction import SklearnFeatureReductionRunner
from face_recognition.infrastructure.persistence.baseline_evaluation_report import BaselineReportError, validate_aggregate_report_target
from face_recognition.infrastructure.persistence.development_dataset_loader import DevelopmentDatasetLoader
from face_recognition.infrastructure.persistence.feature_reduction_report import write_feature_reduction_report
from face_recognition.infrastructure.persistence.identity_split_repository import IdentitySplitPersistenceError, read_identity_split
from face_recognition.infrastructure.persistence.processed_dataset_catalog import ProcessedDatasetReadError


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare full XYZ, ANOVA and PCA on training-pool session folds only")
    parser.add_argument("--processed-root", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--anova-k", action="append", type=int, help="Replace default ANOVA counts; repeat per candidate")
    parser.add_argument("--pca-variance", action="append", type=float, help="Replace default PCA fractions; repeat per candidate")
    args = parser.parse_args(argv)
    try:
        counts = args.anova_k if args.anova_k is not None else [row.k for row in DEFAULT_REDUCTIONS if row.method == "anova"]
        variances = args.pca_variance if args.pca_variance is not None else [
            row.explained_variance for row in DEFAULT_REDUCTIONS if row.method == "pca"
        ]
        try:
            reductions = (ReductionConfiguration("baseline"), *(ReductionConfiguration("anova", k=k) for k in counts),
                          *(ReductionConfiguration("pca", explained_variance=value) for value in variances))
        except ValueError:
            raise FeatureReductionError("Invalid reduction configuration") from None
        evaluator = FeatureReductionEvaluator(SklearnFeatureReductionRunner(), reductions)
        processed, split = args.processed_root.resolve(), args.split_manifest.resolve()
        target = validate_aggregate_report_target(args.report, (processed, split))
        data = DevelopmentDatasetLoader(processed).load(read_identity_split(split))
        report = evaluator.evaluate(data)
        write_feature_reduction_report(report, target, (processed, split))
    except (FeatureReductionError, BaselineReportError, SplitValidationError,
            IdentitySplitPersistenceError, ProcessedDatasetReadError) as error:
        print(f"Feature reduction evaluation failed: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Feature reduction evaluation interrupted.", file=sys.stderr)
        return 1
    except Exception:
        print("Feature reduction evaluation failed: internal or filesystem error.", file=sys.stderr)
        return 1
    print("FEATURE REDUCTION EVALUATION")
    print(f"Source: {report.source_representation}; original features: {report.original_feature_count}; identities: {report.identity_count}")
    for row in report.results:
        evaluation = row.evaluation
        dimensions = ", ".join(f"{fold.fold_name}={fold.transformation.retained_features}" for fold in row.transformations)
        print(f"\n{row.reduction.identifier} / {evaluation.configuration.name}: {dimensions}")
        for fold in evaluation.folds:
            metrics, timing = fold.measurements.metrics, fold.measurements.timing
            print(f"  {fold.fold_name}: accuracy={metrics.accuracy:.6f}, precision={metrics.macro_precision:.6f}, "
                  f"recall={metrics.macro_recall:.6f}, macro F1={metrics.macro_f1:.6f}; "
                  f"fit={timing.fit_seconds:.6f}s, predict={timing.prediction_seconds:.6f}s, "
                  f"batch latency={timing.mean_prediction_latency_ms:.6f}ms/sample")
        print(f"  Mean macro F1={evaluation.mean_metrics.macro_f1:.6f} (std={evaluation.std_metrics.macro_f1:.6f}); "
              f"accuracy={evaluation.mean_metrics.accuracy:.6f}; delta F1={row.comparison.macro_f1_delta:+.6f}")
    lookup = {(row.reduction.identifier, row.evaluation.configuration.name): row for row in report.results}
    print("\nRanked development configurations (exploratory; no final model selected):")
    for ranked in report.ranking:
        row = lookup[ranked.reduction_id, ranked.classifier_name]
        print(f"  {ranked.rank}. {ranked.reduction_id} / {ranked.classifier_name}: "
              f"macro F1={row.evaluation.mean_metrics.macro_f1:.6f}, accuracy={row.evaluation.mean_metrics.accuracy:.6f}, "
              f"mean dimensions={row.mean_retained_features:.2f}, "
              f"batch latency={row.evaluation.mean_timing.mean_prediction_latency_ms:.6f}ms/sample")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
