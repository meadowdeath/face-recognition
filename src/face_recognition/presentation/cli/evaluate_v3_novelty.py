"""Offline V3 nested novelty comparison. Real execution is always explicit."""

import argparse
from pathlib import Path
import sys

from face_recognition.application.evaluate_v3_novelty import V3NoveltyEvaluator
from face_recognition.infrastructure.classifiers.sklearn_v3_novelty import SklearnV3NoveltyRunner
from face_recognition.infrastructure.persistence.aggregate_report import validate_aggregate_report_target
from face_recognition.infrastructure.persistence.artifact_integrity import file_sha256
from face_recognition.infrastructure.persistence.v3_cv_protocol_repository import read_v3_catalog, read_v3_cv_protocol
from face_recognition.infrastructure.persistence.v3_novelty_report import write_v3_novelty_report
from face_recognition.infrastructure.persistence.v3_payload_loader import load_v3_payloads


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fixed-identity V3 nested novelty comparison; outer CV estimates generalization")
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        inputs = (args.processed_root, args.protocol)
        target = validate_aggregate_report_target(args.report, inputs)
        protocol_sha = file_sha256(args.protocol)
        protocol = read_v3_cv_protocol(args.protocol)
        print("V3 NESTED NOVELTY COMPARISON — validating payloads before fitting")
        print(f"Protocol SHA-256: {protocol.protocol_sha256}")
        data = load_v3_payloads(args.processed_root, protocol)
        if file_sha256(args.protocol) != protocol_sha:
            raise ValueError
        result = V3NoveltyEvaluator(SklearnV3NoveltyRunner(), progress=lambda message: print(message, flush=True)).evaluate(data, protocol_sha)
        if (file_sha256(args.protocol) != protocol_sha or read_v3_catalog(args.processed_root) != protocol.catalog
                or any((args.processed_root / name).stat().st_mtime_ns != stamp for name, stamp in data.source_stamps)):
            raise ValueError
        write_v3_novelty_report(result, target, inputs)
    except KeyboardInterrupt:
        print("V3 novelty evaluation interrupted; check for a completed report before retrying.", file=sys.stderr)
        return 1
    except Exception:
        print("V3 novelty evaluation failed: invalid protocol/payload, fitting/scoring inconsistency or existing report. No partial final report published.", file=sys.stderr)
        return 1
    print("V3 OUTER CROSS-VALIDATION GENERALIZATION ESTIMATE")
    print("Fixed: all_xyz_v1/1434 -> ANOVA-448 -> StandardScaler -> RBF SVC C=4 gamma=0.001 probability=False")
    print("Temporal: non-overlapping window=7; median scores; fixed plurality/tie rules; no consistency gate")
    for row in result.outer_results:
        m = row.operational_metrics
        print(f"{row.fold_name}: selected={row.selected_method}; threshold={row.threshold:.12g}; "
              f"known frames/windows={row.known_test_counts.source_samples}/{row.known_test_counts.windows}; "
              f"unknown frames/windows={row.unknown_test_counts.source_samples}/{row.unknown_test_counts.windows}; "
              f"discarded={row.known_test_counts.discarded_samples}/{row.unknown_test_counts.discarded_samples}")
        print(f"  Closed-set accuracy={row.closed_set_known_frames.accuracy:.6f}; "
              f"macro precision={row.closed_set_known_frames.macro_precision:.6f}; macro recall={row.closed_set_known_frames.macro_recall:.6f}; "
              f"macro F1={row.closed_set_known_frames.macro_f1:.6f}; "
              f"AUC={row.diagnostics.roc_auc:.6f}; EER={row.diagnostics.equal_error_point.eer:.6f}; "
              f"FAR={m.unknown_false_acceptance_rate:.6f}; FRR={m.known_rejection_rate:.6f}; BA={m.binary_balanced_accuracy:.6f}")
        print(f"  Known acceptance={m.known_acceptance_rate:.6f}; unknown rejection={m.unknown_rejection_rate:.6f}; "
              f"correct known ID={m.known_correct_identification_rate:.6f}; accepted misidentification={m.known_misidentification_rate:.6f}; "
              f"open accuracy={m.open_set_exact_accuracy:.6f}; binary macro F1={m.binary_macro_f1:.6f}")
    print(f"Outer mean closed-set accuracy={result.mean_closed_set.accuracy:.6f}; macro F1={result.mean_closed_set.macro_f1:.6f}")
    print(f"Outer mean AUC={result.mean_roc_auc:.6f} (population std={result.std_roc_auc:.6f}); "
          f"mean EER={result.mean_eer:.6f}; mean open-set accuracy={result.mean_metrics.open_set_exact_accuracy:.6f}")
    print("Inner candidate comparisons are development-only. V3 has no independent untouched final holdout.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
