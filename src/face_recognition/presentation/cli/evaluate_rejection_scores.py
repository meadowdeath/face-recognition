"""Controlled rejection-score comparison: 0 success, 1 failure, 2 usage error."""

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from face_recognition.application.evaluate_rejection_scores import RejectionScoreEvaluator
from face_recognition.application.unknown_rejection_split import UnknownRejectionError
from face_recognition.application.validate_identity_split import SplitValidationError
from face_recognition.infrastructure.classifiers.sklearn_rejection_scores import SklearnRejectionScoreRunner
from face_recognition.infrastructure.persistence.baseline_evaluation_report import BaselineReportError, validate_aggregate_report_target
from face_recognition.infrastructure.persistence.development_dataset_loader import DevelopmentDatasetLoader
from face_recognition.infrastructure.persistence.identity_split_repository import IdentitySplitPersistenceError, read_identity_split
from face_recognition.infrastructure.persistence.processed_dataset_catalog import ProcessedDatasetReadError
from face_recognition.infrastructure.persistence.rejection_score_report import write_rejection_score_report
from face_recognition.infrastructure.persistence.unknown_calibration_loader import UnknownCalibrationLoader
from face_recognition.infrastructure.persistence.unknown_rejection_repository import read_unknown_split


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare four fixed confidence scores on known development folds and unknown calibration only")
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
        report = RejectionScoreEvaluator(SklearnRejectionScoreRunner()).evaluate(known, unknown)
        write_rejection_score_report(report, target, inputs)
    except (UnknownRejectionError, SplitValidationError, BaselineReportError,
            IdentitySplitPersistenceError, ProcessedDatasetReadError) as error:
        print(f"Rejection-score evaluation failed: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Rejection-score evaluation interrupted.", file=sys.stderr)
        return 1
    except Exception:
        print("Rejection-score evaluation failed: internal or filesystem error.", file=sys.stderr)
        return 1
    print("UNKNOWN REJECTION SCORE EVALUATION")
    print("known_test biometric payloads accessed: no")
    print("unknown_holdout biometric payloads accessed: no")
    print("Higher scores indicate KNOWN; metrics are calibration-set diagnostics at fold-specific BA thresholds.")
    for row in report.results:
        metrics = row.mean_metrics
        print(f"\nANOVA-{row.candidate.k} / SVM / {row.score_family}")
        print(f"  AUC={row.mean_roc_auc:.6f}; EER={row.mean_eer:.6f}; FAR={metrics.unknown_false_acceptance_rate:.6f}; "
              f"FRR={metrics.known_rejection_rate:.6f}")
        print(f"  Known correct ID={metrics.known_correct_identification_rate:.6f}; "
              f"accepted misidentification={metrics.known_misidentification_rate:.6f}; "
              f"open-set accuracy={metrics.open_set_exact_accuracy:.6f}")
        print(f"  Median development threshold={row.median_development_threshold:.12g}; "
              f"threshold gap={row.absolute_threshold_difference:.12g}; AUC delta vs margin={row.vs_margin.roc_auc_delta:+.6f}")
        for fold in row.folds:
            point = fold.analysis.equal_error_point
            print(f"  {fold.fold_name}: AUC={fold.analysis.roc_auc:.6f}; EER={point.eer:.6f} "
                  f"at threshold={point.threshold:.12g}; BA threshold={fold.analysis.calibration.threshold:.12g}")
            for operating in fold.analysis.far_operating_points:
                if operating.metrics is None:
                    print(f"    FAR <= {100 * operating.target_far:g}%: {operating.status}")
                else:
                    m = operating.metrics
                    print(f"    FAR <= {100 * operating.target_far:g}%: {operating.status}; "
                          f"FAR={m.unknown_false_acceptance_rate:.6f}, FRR={m.known_rejection_rate:.6f}, "
                          f"correct ID={m.known_correct_identification_rate:.6f}, "
                          f"open accuracy={m.open_set_exact_accuracy:.6f}, threshold={operating.threshold:.12g}")
    print(f"\nBest rejection-score development candidate: {report.best_rejection_score_development_candidate}")
    print("No final model was selected or persisted; final holdouts remain unevaluated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
