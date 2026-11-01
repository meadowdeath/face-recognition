"""Closed-set development baselines: 0 success, 1 failure, 2 usage error."""

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from face_recognition.application.evaluate_baseline_models import BaselineEvaluationError, BaselineIdentityEvaluator
from face_recognition.application.validate_identity_split import SplitValidationError
from face_recognition.infrastructure.classifiers.sklearn_baselines import SklearnBaselineRunner
from face_recognition.infrastructure.persistence.baseline_evaluation_report import (
    BaselineReportError, validate_baseline_report_target, write_baseline_report,
)
from face_recognition.infrastructure.persistence.development_dataset_loader import DevelopmentDatasetLoader
from face_recognition.infrastructure.persistence.identity_split_repository import IdentitySplitPersistenceError, read_identity_split
from face_recognition.infrastructure.persistence.processed_dataset_catalog import ProcessedDatasetReadError


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate fixed full-XYZ identity baselines on training-pool session folds")
    parser.add_argument("--processed-root", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path, help="New external aggregate JSON report; never overwritten")
    args = parser.parse_args(argv)
    try:
        processed, split = args.processed_root.resolve(), args.split_manifest.resolve()
        target = validate_baseline_report_target(args.report, (processed, split))
        manifest = read_identity_split(split)
        data = DevelopmentDatasetLoader(processed).load(manifest)
        report = BaselineIdentityEvaluator(SklearnBaselineRunner()).evaluate(data)
        write_baseline_report(report, target, (processed, split))
    except (BaselineEvaluationError, BaselineReportError, SplitValidationError,
            IdentitySplitPersistenceError, ProcessedDatasetReadError) as error:
        print(f"Baseline evaluation failed: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Baseline evaluation interrupted.", file=sys.stderr)
        return 1
    except Exception:
        print("Baseline evaluation failed: internal or filesystem error.", file=sys.stderr)
        return 1
    print("BASELINE IDENTITY EVALUATION")
    print(f"Representation: {report.representation}")
    print(f"Features: {report.feature_count}; known identities: {report.identity_count}")
    print("Development partition: known_train_pool; unweighted fold means; population std.")
    for classifier in report.classifiers:
        print(f"\n{classifier.configuration.name}")
        for fold in classifier.folds:
            metrics, timing = fold.measurements.metrics, fold.measurements.timing
            print(f"  {fold.fold_name}: fit={fold.fit_samples}, validation={fold.validation_samples}")
            print(f"    Accuracy: {metrics.accuracy:.6f}; macro precision: {metrics.macro_precision:.6f}; "
                  f"macro recall: {metrics.macro_recall:.6f}; macro F1: {metrics.macro_f1:.6f}")
            print(f"    Fit: {timing.fit_seconds:.6f}s; prediction: {timing.prediction_seconds:.6f}s; "
                  f"batch mean latency: {timing.mean_prediction_latency_ms:.6f}ms/sample")
        for name in classifier.mean_metrics.__dataclass_fields__:
            print(f"  Mean {name}: {getattr(classifier.mean_metrics, name):.6f}; "
                  f"std: {getattr(classifier.std_metrics, name):.6f}")
        timing = classifier.mean_timing
        print(f"  Mean fit: {timing.fit_seconds:.6f}s; mean prediction: {timing.prediction_seconds:.6f}s; "
              f"mean batch latency: {timing.mean_prediction_latency_ms:.6f}ms/sample")
    print("\nExternal aggregate report created. No protected partition payloads were read.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
