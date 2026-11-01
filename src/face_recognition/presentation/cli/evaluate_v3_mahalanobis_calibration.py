"""V3 calibration refinement only; one fitted model per outer fold."""

import argparse
from pathlib import Path
import sys

from face_recognition.application.evaluate_v3_mahalanobis_calibration import V3MahalanobisCalibrationEvaluator
from face_recognition.infrastructure.classifiers.sklearn_v3_same_model import SklearnV3SameModelRunner
from face_recognition.infrastructure.persistence.aggregate_report import validate_aggregate_report_target
from face_recognition.infrastructure.persistence.artifact_integrity import file_sha256
from face_recognition.infrastructure.persistence.v3_cv_protocol_repository import read_v3_catalog, read_v3_cv_protocol
from face_recognition.infrastructure.persistence.v3_mahalanobis_calibration_report import (
    read_mahalanobis_selection_report, write_mahalanobis_calibration_report,
)
from face_recognition.infrastructure.persistence.v3_payload_loader import load_v3_payloads


def main(argv=None):
    parser = argparse.ArgumentParser(description="V3 threshold-calibration refinement under CV, not independent final evaluation")
    for name in ("processed-root", "protocol", "selection-report", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        inputs = (args.processed_root, args.protocol, args.selection_report)
        target = validate_aggregate_report_target(args.report, inputs)
        protocol_sha = file_sha256(args.protocol)
        protocol = read_v3_cv_protocol(args.protocol)
        selection = read_mahalanobis_selection_report(args.selection_report, protocol)
        print("V3 THRESHOLD-CALIBRATION REFINEMENT UNDER CROSS-VALIDATION", flush=True)
        print(f"Protocol SHA-256: {protocol.protocol_sha256}; method=mahalanobis_distance", flush=True)
        data = load_v3_payloads(args.processed_root, protocol)
        result = V3MahalanobisCalibrationEvaluator(SklearnV3SameModelRunner(),
                    progress=lambda message: print(message, flush=True)).evaluate(data, protocol_sha, selection)
        if (file_sha256(args.protocol) != protocol_sha or file_sha256(args.selection_report) != selection.report_sha256
                or read_v3_catalog(args.processed_root) != protocol.catalog
                or any((args.processed_root / name).stat().st_mtime_ns != stamp for name, stamp in data.source_stamps)):
            raise ValueError
        write_mahalanobis_calibration_report(result, target, inputs)
    except KeyboardInterrupt:
        print("Calibration study interrupted; check for a completed report before retrying.", file=sys.stderr)
        return 1
    except Exception:
        print("Calibration study failed: invalid provenance/payload, changed fitted model or existing output. No partial final report published.", file=sys.stderr)
        return 1
    print("Fixed: all_xyz_v1/1434 -> ANOVA-448 -> StandardScaler -> RBF SVC C=4 gamma=0.001 probability=False")
    print("Mahalanobis; non-overlapping window=7; median confidence; existing plurality/ties; no consistency gate")
    for row in result.outer_results:
        c, m, closed = row.calibration_metrics, row.operational_metrics, row.closed_set_known_frames
        print(f"{row.roles.fold_name}: fit={','.join(row.roles.known_fit_sessions)}; calibration={row.roles.known_calibration_session}; "
              f"known test={row.roles.known_test_session}; calibration UNKNOWN identities={len(row.roles.unknown_calibration_ids)}; threshold={row.threshold:.12g}")
        print(f"  Calibration: FAR={c.unknown_false_acceptance_rate:.6f}; FRR={c.known_rejection_rate:.6f}; "
              f"BA={c.binary_balanced_accuracy:.6f}; AUC={row.calibration_diagnostics.roc_auc:.6f}; EER={row.calibration_diagnostics.equal_error_point.eer:.6f}")
        print(f"  Calibration frames/windows: known={row.known_calibration_counts.source_samples}/{row.known_calibration_counts.windows}; "
              f"unknown={row.unknown_calibration_counts.source_samples}/{row.unknown_calibration_counts.windows}; "
              f"discarded={row.known_calibration_counts.discarded_samples}/{row.unknown_calibration_counts.discarded_samples}")
        print(f"  Outer closed-set: accuracy={closed.accuracy:.6f}; macro precision={closed.macro_precision:.6f}; "
              f"macro recall={closed.macro_recall:.6f}; macro F1={closed.macro_f1:.6f}")
        print(f"  Outer: FAR={m.unknown_false_acceptance_rate:.6f}; FRR={m.known_rejection_rate:.6f}; "
              f"known acceptance={m.known_acceptance_rate:.6f}; unknown rejection={m.unknown_rejection_rate:.6f}; "
              f"correct known ID={m.known_correct_identification_rate:.6f}; accepted misidentification={m.known_misidentification_rate:.6f}; "
              f"open accuracy={m.open_set_exact_accuracy:.6f}; BA={m.binary_balanced_accuracy:.6f}; binary macro F1={m.binary_macro_f1:.6f}")
        print(f"  Outer diagnostic AUC={row.diagnostics.roc_auc:.6f}; EER={row.diagnostics.equal_error_point.eer:.6f}; "
              f"known frames/windows={row.known_test_counts.source_samples}/{row.known_test_counts.windows}; "
              f"unknown frames/windows={row.unknown_test_counts.source_samples}/{row.unknown_test_counts.windows}; "
              f"discarded={row.known_test_counts.discarded_samples}/{row.unknown_test_counts.discarded_samples}")
    print(f"Mean/std closed accuracy={result.mean_closed_set.accuracy:.6f}/{result.std_closed_set.accuracy:.6f}; "
          f"macro F1={result.mean_closed_set.macro_f1:.6f}/{result.std_closed_set.macro_f1:.6f}; "
          f"AUC={result.mean_roc_auc:.6f}/{result.std_roc_auc:.6f}; EER={result.mean_eer:.6f}/{result.std_eer:.6f}")
    print(f"FAR range={result.far_range[0]:.6f}..{result.far_range[1]:.6f}; FRR range={result.frr_range[0]:.6f}..{result.frr_range[1]:.6f}")
    print(f"Mean/std FAR={result.mean_metrics.unknown_false_acceptance_rate:.6f}/{result.std_metrics.unknown_false_acceptance_rate:.6f}; "
          f"FRR={result.mean_metrics.known_rejection_rate:.6f}/{result.std_metrics.known_rejection_rate:.6f}; "
          f"BA={result.mean_metrics.binary_balanced_accuracy:.6f}/{result.std_metrics.binary_balanced_accuracy:.6f}; "
          f"open accuracy={result.mean_metrics.open_set_exact_accuracy:.6f}/{result.std_metrics.open_set_exact_accuracy:.6f}")
    print("threshold_transport=none_same_model_calibration; no refit after calibration; original selection report unchanged")
    print("Prior outer results motivated this refinement. V3 has no additional untouched holdout; V2 remains the last independent external holdout.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
