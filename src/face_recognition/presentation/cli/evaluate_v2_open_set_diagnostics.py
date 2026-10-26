"""V2 development diagnostics: 0 success, 1 failure, 2 usage error.

V1's aggregate report is a read-only reference. No V1 model artifact is loaded.
"""

import argparse
from hashlib import sha256
from pathlib import Path
import sys

from face_recognition.application.evaluate_v2_open_set_diagnostics import V2DiagnosticEvaluator
from face_recognition.infrastructure.classifiers.sklearn_v2_diagnostics import SklearnV2DiagnosticRunner
from face_recognition.infrastructure.persistence.baseline_evaluation_report import validate_aggregate_report_target
from face_recognition.infrastructure.persistence.identity_split_repository import read_identity_split
from face_recognition.infrastructure.persistence.unknown_rejection_repository import read_unknown_split
from face_recognition.infrastructure.persistence.v2_diagnostics_report import write_v2_report
from face_recognition.infrastructure.persistence.v2_study_loader import V2StudyLoader, read_v1_reference


def main(argv=None):
    parser = argparse.ArgumentParser(description="V2 DEVELOPMENT diagnostics on original V1 data only; V1 remains immutable")
    for name in ("processed-root", "identity-split", "unknown-split", "v1-final-report", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        inputs = tuple(p.resolve() for p in (args.processed_root, args.identity_split, args.unknown_split, args.v1_final_report))
        target = validate_aggregate_report_target(args.report, inputs)
        source = read_identity_split(inputs[1])
        partition = read_unknown_split(inputs[2], source)
        baseline = read_v1_reference(inputs[3], source, partition)
        data = V2StudyLoader(inputs[0]).load_study(source, partition, baseline)
        report = V2DiagnosticEvaluator().evaluate(data, baseline, SklearnV2DiagnosticRunner())
        if sha256(inputs[3].read_bytes()).hexdigest() != baseline.report_sha256:
            raise ValueError("V1 reference changed during evaluation")
        write_v2_report(report, target, inputs)
    except KeyboardInterrupt:
        print("V2 study interrupted; no completed report published.", file=sys.stderr)
        return 1
    except Exception:
        # JSON/sklearn errors can contain input values. Never echo their payload.
        print("V2 study failed: invalid source/reference, incompatible original catalog, analysis or output error. No V1 changes permitted.", file=sys.stderr)
        return 1
    print("V2 OPEN-SET DIAGNOSTIC STUDY — DEVELOPMENT ONLY")
    print(f"V1 baseline: accuracy={baseline.closed_set_accuracy:.6f}; FAR={baseline.far:.6f}; FRR={baseline.frr:.6f}; open accuracy={baseline.open_set_accuracy:.6f}")
    print(f"Known out-of-session frames: accuracy={report.closed_set_frame_oof.metrics.accuracy:.6f}; macro F1={report.closed_set_frame_oof.metrics.macro_f1:.6f}")
    print(f"Three known-session fits; {report.unknown_development_identity_count} unknown identity rotations per fold; 42 configurations.")
    rows = {r["identifier"]: r for r in report.candidates}
    print("Rejection / temporal comparison (AUC describes raw median score, not identity thresholds/gate):")
    for row in report.candidates:
        c, m = row["configuration"], row["metrics"]
        print(f"  {c['strategy']} window={c['window']} gate={c['strict_majority_gate']}: AUC={row['mean_roc_auc']:.6f}; EER={row['mean_eer']:.6f}; FAR={m['unknown_false_acceptance_rate']:.6f}; FRR={m['known_rejection_rate']:.6f}; open={m['open_set_exact_accuracy']:.6f}")
    winner = rows[report.best_v2_development_candidate]
    print(f"Best V2 development candidate: {report.best_v2_development_candidate}")
    print("Worst known identities by FRR (selected development candidate):")
    for row in sorted((r for r in winner["known_identity_diagnostics"] if r["known_rejection_rate"] is not None),
                      key=lambda r: (-r["known_rejection_rate"], r["person_id"]))[:3]:
        print(f"  {row['person_id']}: FRR={row['known_rejection_rate']:.6f}")
    print("Worst known conditions by FRR:")
    for row in sorted((r for r in winner["known_condition_diagnostics"] if r["known_rejection_rate"] is not None),
                      key=lambda r: (-r["known_rejection_rate"], r["condition"]))[:3]:
        print(f"  {row['condition']}: FRR={row['known_rejection_rate']:.6f}")
    print("V1 report unchanged; no V1 trained artifacts loaded or modified.")
    print("V2 IS NOT FROZEN. No production model persisted.")
    print("Freeze V2 before collecting/evaluating a NEW known session (conceptually session_04) and NEW unknown identities.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
