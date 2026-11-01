"""Development-only calibration: 0 success, 1 failure, 2 usage error."""

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from face_recognition.application.calibrate_unknown_rejection import UnknownRejectionCalibrator
from face_recognition.application.unknown_rejection_split import UnknownRejectionError
from face_recognition.application.validate_identity_split import SplitValidationError
from face_recognition.infrastructure.classifiers.sklearn_unknown_rejection import SklearnUnknownRejectionRunner
from face_recognition.infrastructure.persistence.baseline_evaluation_report import BaselineReportError, validate_aggregate_report_target
from face_recognition.infrastructure.persistence.development_dataset_loader import DevelopmentDatasetLoader
from face_recognition.infrastructure.persistence.identity_split_repository import IdentitySplitPersistenceError, read_identity_split
from face_recognition.infrastructure.persistence.processed_dataset_catalog import ProcessedDatasetReadError
from face_recognition.infrastructure.persistence.unknown_calibration_loader import UnknownCalibrationLoader
from face_recognition.infrastructure.persistence.unknown_rejection_repository import read_unknown_split, write_rejection_report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
        report = UnknownRejectionCalibrator(SklearnUnknownRejectionRunner()).evaluate(known, unknown)
        write_rejection_report(report, target, inputs)
    except (UnknownRejectionError, SplitValidationError, BaselineReportError,
            IdentitySplitPersistenceError, ProcessedDatasetReadError) as error:
        print(f"Unknown rejection calibration failed: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Unknown rejection calibration interrupted.", file=sys.stderr)
        return 1
    except Exception:
        print("Unknown rejection calibration failed: internal or filesystem error.", file=sys.stderr)
        return 1
    print("UNKNOWN REJECTION CALIBRATION")
    print(f"Unknown calibration identities: {partition.calibration.identities}; samples: {partition.calibration.samples}")
    print(f"Unknown holdout identities: {partition.holdout.identities}; samples: {partition.holdout.samples}")
    print("Holdout biometric payloads accessed: no")
    print("Metrics below use fold-specific calibrated thresholds; median-threshold metrics are also stored in the report.")
    for row in report.results:
        print(f"\nANOVA-{row.candidate.k} / SVM C={row.candidate.C:g}, gamma={row.candidate.gamma}")
        for fold in row.folds:
            print(f"  {fold.fold_name} threshold: {fold.calibration.threshold:.12g}")
        print(f"  Development threshold: {row.development_rejection_threshold:.12g}; "
              f"absolute threshold difference: {row.absolute_threshold_difference:.12g}")
        metrics = row.mean_metrics
        print(f"  Known correct identification: {metrics.known_correct_identification_rate:.6f}; "
              f"known rejection / FRR: {metrics.known_rejection_rate:.6f}")
        print(f"  Unknown rejection: {metrics.unknown_rejection_rate:.6f}; "
              f"unknown false acceptance / FAR: {metrics.unknown_false_acceptance_rate:.6f}")
        print(f"  Open-set accuracy: {metrics.open_set_exact_accuracy:.6f}; "
              f"binary balanced accuracy: {metrics.binary_balanced_accuracy:.6f}; "
              f"closed-set macro F1: {row.mean_closed_set_metrics.macro_f1:.6f}")
    print(f"\nSelected open-set development candidate: {report.selected_open_set_development_candidate.identifier}")
    print("No final model was trained. Margin/threshold transfer after refitting requires future holdout evaluation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
