"""Controlled representation comparison; no hyperparameter or novelty search."""

import argparse
from pathlib import Path
import sys

from face_recognition.application.evaluate_v3_pose_representation import V3PoseRepresentationEvaluator
from face_recognition.domain.models.processed_sample import POSE_REPRESENTATION
from face_recognition.infrastructure.classifiers.sklearn_v3_same_model import SklearnV3SameModelRunner
from face_recognition.infrastructure.features.pose_landmark_normalizer import POSE_NORMALIZATION
from face_recognition.infrastructure.persistence.aggregate_report import validate_aggregate_report_target
from face_recognition.infrastructure.persistence.artifact_integrity import file_sha256
from face_recognition.infrastructure.persistence.v3_cv_protocol_repository import read_v3_catalog, read_v3_cv_protocol
from face_recognition.infrastructure.persistence.v3_pose_representation import (
    load_paired_pose_payloads, pose_processing_fingerprints, write_pose_representation_report,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Iterative V3 baseline vs face-local pose representation, same-model Mahalanobis")
    for name in ("baseline-root", "candidate-root", "protocol", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        inputs = (args.baseline_root, args.candidate_root, args.protocol)
        target = validate_aggregate_report_target(args.report, inputs)
        protocol_sha = file_sha256(args.protocol)
        protocol = read_v3_cv_protocol(args.protocol)
        print("V3 POSE REPRESENTATION COMPARISON — iterative study, not an independent final holdout", flush=True)
        baseline, candidate, candidate_catalog, manifests = load_paired_pose_payloads(args.baseline_root, args.candidate_root, protocol)
        result = V3PoseRepresentationEvaluator(SklearnV3SameModelRunner,
            progress=lambda message: print(message, flush=True)).evaluate(baseline, candidate, protocol_sha, candidate_catalog, POSE_NORMALIZATION)
        if (file_sha256(args.protocol) != protocol_sha or read_v3_catalog(args.baseline_root) != protocol.catalog
                or read_v3_catalog(args.candidate_root, representation=POSE_REPRESENTATION) != candidate_catalog
                or pose_processing_fingerprints(args.candidate_root) != manifests
                or any((root / path).stat().st_mtime_ns != stamp for root, data in
                    ((args.baseline_root, baseline), (args.candidate_root, candidate)) for path, stamp in data.source_stamps)):
            raise ValueError
        write_pose_representation_report(result, target, inputs)
    except KeyboardInterrupt:
        print("Comparison interrupted; check for a completed report before retrying.", file=sys.stderr)
        return 1
    except Exception:
        print("Representation comparison failed: invalid inputs/provenance, fitting or output target. No partial final report published.", file=sys.stderr)
        return 1
    for row in result.results:
        m, c = row.mean_metrics, row.mean_closed_set
        print(f"{row.representation}: closed-set accuracy={c.accuracy:.6f}, precision={c.macro_precision:.6f}, recall={c.macro_recall:.6f}, F1={c.macro_f1:.6f}")
        print(f"  AUC={row.mean_roc_auc:.6f}, EER={row.mean_eer:.6f}, FAR={m.unknown_false_acceptance_rate:.6f}, FRR={m.known_rejection_rate:.6f}")
        print(f"  known acceptance={m.known_acceptance_rate:.6f}, unknown rejection={m.unknown_rejection_rate:.6f}")
        print(f"  correct ID={m.known_correct_identification_rate:.6f}, accepted misID={m.known_misidentification_rate:.6f}, open-set accuracy={m.open_set_exact_accuracy:.6f}")
        print(f"  balanced accuracy={m.binary_balanced_accuracy:.6f}, binary macro F1={m.binary_macro_f1:.6f}")
    print("KNOWN condition FRR: baseline / candidate / delta(candidate-baseline)")
    for row in result.condition_comparison:
        print(f"  {row.condition}: {row.baseline_frr:.6f} / {row.candidate_frr:.6f} / {row.frr_delta_candidate_minus_baseline:+.6f}")
    print(f"Preferred representation in this iterative study: {result.selected_development_representation}")
    print("Detailed score summaries, UNKNOWN condition/identity FAR and KNOWN session FRR are in the aggregate JSON report.")
    print("V2 remains the last independent external holdout. No real independent final generalization claim.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
