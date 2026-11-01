"""Temporal nearest-known development evaluation: 0 success, 1 failure, 2 usage."""

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from face_recognition.application.evaluate_temporal_distance_novelty import TemporalDistanceNoveltyEvaluator
from face_recognition.application.unknown_rejection_split import UnknownRejectionError
from face_recognition.application.validate_identity_split import SplitValidationError
from face_recognition.infrastructure.classifiers.sklearn_temporal_distance_novelty import SklearnTemporalDistanceNoveltyRunner
from face_recognition.infrastructure.persistence.baseline_evaluation_report import BaselineReportError, validate_aggregate_report_target
from face_recognition.infrastructure.persistence.development_dataset_loader import DevelopmentDatasetLoader
from face_recognition.infrastructure.persistence.identity_split_repository import IdentitySplitPersistenceError, read_identity_split
from face_recognition.infrastructure.persistence.processed_dataset_catalog import ProcessedDatasetReadError
from face_recognition.infrastructure.persistence.temporal_distance_novelty_report import write_temporal_distance_report
from face_recognition.infrastructure.persistence.unknown_calibration_loader import UnknownCalibrationLoader
from face_recognition.infrastructure.persistence.unknown_rejection_repository import read_unknown_split


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate fixed ANOVA-448/SVM with temporal nearest_1 and mean_5nn novelty")
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
        report = TemporalDistanceNoveltyEvaluator(SklearnTemporalDistanceNoveltyRunner()).evaluate(known, unknown)
        write_temporal_distance_report(report, target, inputs)
    except (UnknownRejectionError, SplitValidationError, BaselineReportError,
            IdentitySplitPersistenceError, ProcessedDatasetReadError) as error:
        print(f"Temporal distance novelty evaluation failed: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Temporal distance novelty evaluation interrupted.", file=sys.stderr)
        return 1
    except Exception:
        print("Temporal distance novelty evaluation failed: internal or filesystem error.", file=sys.stderr)
        return 1
    print("TEMPORAL DISTANCE-NOVELTY EVALUATION")
    print("Model: ANOVA-448 / StandardScaler / RBF SVM C=4 / gamma=0.001")
    print("known_test biometric payloads accessed: no")
    print("unknown_holdout biometric payloads accessed: no")
    print("Counts are fold observations; unknown calibration windows are scored by both fold models.")
    for row in report.results:
        m, delta = row.mean_metrics, row.vs_window1
        print(f"\n{row.score_family}, window={row.window_size}: AUC={row.mean_roc_auc:.6f}; EER={row.mean_eer:.6f}; "
              f"FAR={m.unknown_false_acceptance_rate:.6f}; FRR={m.known_rejection_rate:.6f}")
        print(f"  Known correct ID={m.known_correct_identification_rate:.6f}; "
              f"accepted misidentification={m.known_misidentification_rate:.6f}; open accuracy={m.open_set_exact_accuracy:.6f}")
        print(f"  vs SAME method window=1: AUC={delta.roc_auc_delta:+.6f}, EER={delta.eer_delta:+.6f}, "
              f"FAR={delta.far_delta:+.6f}, FRR={delta.frr_delta:+.6f}, "
              f"correct ID={delta.known_correct_identification_delta:+.6f}, open accuracy={delta.open_set_accuracy_delta:+.6f}")
        for side, counts in (("Known", row.total_known_counts), ("Unknown", row.total_unknown_counts)):
            print(f"  {side}: windows={counts.windows}; used={counts.used_samples}/{counts.source_samples}; "
                  f"discarded={counts.discarded_samples}")
        for fold in row.folds:
            points = "; ".join(f"FAR<={100 * p.target_far:g}% {p.status} "
                               f"(FAR={p.metrics.unknown_false_acceptance_rate:.4f}, FRR={p.metrics.known_rejection_rate:.4f})"
                               if p.metrics else f"FAR<={100 * p.target_far:g}% unavailable"
                               for p in fold.analysis.far_operating_points)
            print(f"  {fold.fold_name}: {points}")
    print(f"\nBest temporal distance-novelty development candidate: {report.best_temporal_distance_novelty_development_candidate}")
    print("Larger windows need not improve performance; inspect deltas and discarded counts. No final model was selected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
