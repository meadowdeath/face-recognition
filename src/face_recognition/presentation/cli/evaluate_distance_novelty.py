"""Frame-level novelty diagnostics: 0 success, 1 failure/interruption, 2 usage."""

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from face_recognition.application.evaluate_distance_novelty import DistanceNoveltyEvaluator
from face_recognition.application.unknown_rejection_split import UnknownRejectionError
from face_recognition.application.validate_identity_split import SplitValidationError
from face_recognition.infrastructure.classifiers.sklearn_distance_novelty import SklearnDistanceNoveltyRunner
from face_recognition.infrastructure.persistence.baseline_evaluation_report import BaselineReportError, validate_aggregate_report_target
from face_recognition.infrastructure.persistence.development_dataset_loader import DevelopmentDatasetLoader
from face_recognition.infrastructure.persistence.distance_novelty_report import write_distance_novelty_report
from face_recognition.infrastructure.persistence.identity_split_repository import IdentitySplitPersistenceError, read_identity_split
from face_recognition.infrastructure.persistence.processed_dataset_catalog import ProcessedDatasetReadError
from face_recognition.infrastructure.persistence.unknown_calibration_loader import UnknownCalibrationLoader
from face_recognition.infrastructure.persistence.unknown_rejection_repository import read_unknown_split


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate known-fit 1/3/5-neighbor Euclidean novelty with fixed ANOVA-448/SVM")
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
        report = DistanceNoveltyEvaluator(SklearnDistanceNoveltyRunner()).evaluate(known, unknown)
        write_distance_novelty_report(report, target, inputs)
    except (UnknownRejectionError, SplitValidationError, BaselineReportError,
            IdentitySplitPersistenceError, ProcessedDatasetReadError) as error:
        print(f"Distance novelty evaluation failed: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Distance novelty evaluation interrupted.", file=sys.stderr)
        return 1
    except Exception:
        print("Distance novelty evaluation failed: internal or filesystem error.", file=sys.stderr)
        return 1
    print("DISTANCE-BASED NOVELTY EVALUATION")
    print("Model: ANOVA-448 / StandardScaler / RBF SVM C=4 / gamma=0.001")
    print("known_test biometric payloads accessed: no")
    print("unknown_holdout biometric payloads accessed: no")
    print("Frame-level calibration diagnostics; higher score means KNOWN. Identity still comes from SVM.")
    for title, results in (("Novelty", report.novelty_results), ("Reference (same fits)", report.reference_results)):
        print(f"\n{title}:")
        for row in results:
            m = row.mean_metrics
            print(f"  {row.score_family}: AUC={row.mean_roc_auc:.6f}; EER={row.mean_eer:.6f}; "
                  f"FAR={m.unknown_false_acceptance_rate:.6f}; FRR={m.known_rejection_rate:.6f}")
            print(f"    Known correct ID={m.known_correct_identification_rate:.6f}; "
                  f"accepted misidentification={m.known_misidentification_rate:.6f}; open accuracy={m.open_set_exact_accuracy:.6f}")
            print(f"    Median development threshold={row.median_development_threshold:.12g}; "
                  f"threshold gap={row.absolute_threshold_difference:.12g}")
            for fold in row.folds:
                points = "; ".join(f"FAR<={100 * point.target_far:g}% {point.status} "
                                   f"(FAR={point.metrics.unknown_false_acceptance_rate:.4f}, FRR={point.metrics.known_rejection_rate:.4f})"
                                   if point.metrics is not None else f"FAR<={100 * point.target_far:g}% unavailable"
                                   for point in fold.analysis.far_operating_points)
                print(f"    {fold.fold_name}: {points}")
    print(f"\nNearest same-identity rate (known validation): {100 * report.mean_nearest_same_identity_rate:.2f}%")
    print(f"Unknown nearest-label majority concentration: {100 * report.mean_unknown_nearest_majority_concentration:.2f}% (not identity truth)")
    print(f"Best distance-based novelty development candidate: {report.best_distance_based_novelty_development_candidate}")
    print("No final model was selected or persisted; protected holdouts remain unevaluated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
