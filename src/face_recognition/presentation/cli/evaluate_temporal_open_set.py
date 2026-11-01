"""Offline temporal open-set diagnostics: 0 success, 1 failure, 2 usage error."""

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from face_recognition.application.evaluate_temporal_open_set import TemporalOpenSetEvaluator
from face_recognition.application.unknown_rejection_split import UnknownRejectionError
from face_recognition.application.validate_identity_split import SplitValidationError
from face_recognition.infrastructure.classifiers.sklearn_temporal_open_set import SklearnTemporalOpenSetRunner
from face_recognition.infrastructure.persistence.baseline_evaluation_report import BaselineReportError, validate_aggregate_report_target
from face_recognition.infrastructure.persistence.development_dataset_loader import DevelopmentDatasetLoader
from face_recognition.infrastructure.persistence.identity_split_repository import IdentitySplitPersistenceError, read_identity_split
from face_recognition.infrastructure.persistence.processed_dataset_catalog import ProcessedDatasetReadError
from face_recognition.infrastructure.persistence.temporal_open_set_report import write_temporal_report
from face_recognition.infrastructure.persistence.unknown_calibration_loader import UnknownCalibrationLoader
from face_recognition.infrastructure.persistence.unknown_rejection_repository import read_unknown_split


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate fixed ANOVA-448/SVM with non-overlapping 1/3/5/7-result windows")
    parser.add_argument("--processed-root", required=True, type=Path)
    parser.add_argument("--identity-split", required=True, type=Path)
    parser.add_argument("--unknown-split", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        processed, source_path, partition_path = args.processed_root.resolve(), args.identity_split.resolve(), args.unknown_split.resolve()
        inputs = (processed, source_path, partition_path)
        target = validate_aggregate_report_target(args.report, inputs)
        source = read_identity_split(source_path)
        partition = read_unknown_split(partition_path, source)
        known = DevelopmentDatasetLoader(processed).load(source)
        unknown = UnknownCalibrationLoader(processed).load_calibration(source, partition, known.X.shape[1])
        report = TemporalOpenSetEvaluator(SklearnTemporalOpenSetRunner()).evaluate(known, unknown)
        write_temporal_report(report, target, inputs)
    except (UnknownRejectionError, SplitValidationError, BaselineReportError,
            IdentitySplitPersistenceError, ProcessedDatasetReadError) as error:
        print(f"Temporal evaluation failed: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Temporal evaluation interrupted.", file=sys.stderr)
        return 1
    except Exception:
        print("Temporal evaluation failed: internal or filesystem error.", file=sys.stderr)
        return 1
    print("TEMPORAL OPEN-SET EVALUATION")
    print("Model: ANOVA-448 / RBF SVM C=4 / gamma=0.001")
    print("known_test biometric payloads accessed: no")
    print("unknown_holdout biometric payloads accessed: no")
    print("Counts below are fold observations; unknown calibration samples are reused across the two fold models.")
    for row in report.results:
        m = row.mean_metrics
        print(f"\n{row.score_family}, window={row.window_size}: AUC={row.mean_roc_auc:.6f}; EER={row.mean_eer:.6f}; "
              f"FAR={m.unknown_false_acceptance_rate:.6f}; FRR={m.known_rejection_rate:.6f}")
        print(f"  Known correct ID={m.known_correct_identification_rate:.6f}; "
              f"accepted misidentification={m.known_misidentification_rate:.6f}; open accuracy={m.open_set_exact_accuracy:.6f}; "
              f"binary BA={m.binary_balanced_accuracy:.6f}")
        print(f"  Known windows={row.total_known_counts.windows}, used={row.total_known_counts.used_samples}, "
              f"discarded={row.total_known_counts.discarded_samples}; unknown windows={row.total_unknown_counts.windows}, "
              f"used={row.total_unknown_counts.used_samples}, discarded={row.total_unknown_counts.discarded_samples}")
        print(f"  Delta vs window=1: AUC={row.vs_window1.roc_auc_delta:+.6f}, EER={row.vs_window1.eer_delta:+.6f}, "
              f"FAR={row.vs_window1.far_delta:+.6f}, open accuracy={row.vs_window1.open_set_accuracy_delta:+.6f}")
        for fold in row.folds:
            print(f"  {fold.fold_name}: BA threshold={fold.analysis.calibration.threshold:.12g}; "
                  f"EER threshold={fold.analysis.equal_error_point.threshold:.12g}")
            points = "; ".join(f"FAR<={100 * point.target_far:g}% {point.status} "
                               f"(actual={point.metrics.unknown_false_acceptance_rate:.4f}, "
                               f"FRR={point.metrics.known_rejection_rate:.4f})" if point.metrics is not None
                               else f"FAR<={100 * point.target_far:g}% unavailable"
                               for point in fold.analysis.far_operating_points)
            print(f"    {points}")
    print(f"\nBest temporal open-set development candidate: {report.best_temporal_open_set_development_candidate}")
    print("Inspect deltas and discarded counts; larger windows are not automatically better. No final model was selected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
