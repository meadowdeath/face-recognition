"""One-time independent V2 final evaluation, no parameter overrides or calibration."""

import argparse
from pathlib import Path
import sys

from face_recognition.application.evaluate_final_v2_open_set import FinalV2OpenSetEvaluator
from face_recognition.domain.models.final_v2_evaluation import FinalV2Context, V2_NO_RETUNING
from face_recognition.infrastructure.classifiers.sklearn_final_v2_open_set import SklearnFinalV2OpenSetRunner
from face_recognition.infrastructure.persistence.baseline_evaluation_report import validate_aggregate_report_target
from face_recognition.infrastructure.persistence.final_v2_evaluation_report import write_final_v2_report
from face_recognition.infrastructure.persistence.final_v2_holdout_repository import FinalV2HoldoutLoader, read_final_v2_manifest, read_v2_final_inputs
from face_recognition.infrastructure.persistence.v2_model_repository import load_v2_open_set_model, v2_model_fingerprints


def main(argv=None):
    parser = argparse.ArgumentParser(description="ONE-TIME independent final V2 evaluation; never use results to retune this version")
    for name in ("processed-root", "identity-split", "unknown-split", "final-holdout", "model-dir", "frozen-config", "development-report", "report"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        roots = tuple(p.resolve() for p in (args.processed_root, args.identity_split, args.unknown_split, args.final_holdout,
                      args.model_dir, args.frozen_config, args.development_report))
        target = validate_aggregate_report_target(args.report, roots)  # Refuse repeat BEFORE any holdout access.
        inputs = read_v2_final_inputs(args.identity_split, args.unknown_split, args.frozen_config, args.development_report, args.model_dir)
        manifest, final_sha = read_final_v2_manifest(args.final_holdout)
        loader = FinalV2HoldoutLoader(args.processed_root)
        loader.authorize(manifest, inputs)  # Catalog only, before feature records open.
        model = load_v2_open_set_model(args.model_dir, inputs.config, inputs.config_sha256)
        before = v2_model_fingerprints(args.model_dir)  # Immediately before final feature access.
        if before != inputs.model_fingerprints or model.parts.class_ids != tuple(inputs.model_metadata["class_ids"]):
            raise ValueError
        context = FinalV2Context(inputs.config, inputs.config_sha256, inputs.config.provenance.sha256,
                    inputs.config.provenance.original_identity_split_sha256, inputs.unknown_split_sha256, final_sha,
                    manifest.processing_manifest_sha256, before, model.threshold, model.manifest["calibration_analysis"]["roc_auc"],
                    model.manifest["calibration_analysis"]["calibration"]["metrics"]["binary_balanced_accuracy"])
        data = loader.load_final(manifest, inputs)
        result = FinalV2OpenSetEvaluator(SklearnFinalV2OpenSetRunner(model)).evaluate(data, context)
        if v2_model_fingerprints(args.model_dir) != before:
            raise ValueError
        write_final_v2_report(result, target, roots, args.model_dir)
    except KeyboardInterrupt:
        print("Final V2 evaluation interrupted; check for a completed report before retrying. Do not retune.", file=sys.stderr)
        return 1
    except Exception:
        print("Final V2 evaluation failed: invalid provenance/independence, incomplete catalog/records, changed model or existing report. No fitting or recalibration performed.", file=sys.stderr)
        return 1
    a, m, closed = inputs.config.architecture, result.operational_metrics, result.closed_set_known_frames
    print("FINAL V2 OPEN-SET HOLDOUT EVALUATION — INDEPENDENT GENERALIZATION")
    print(f"Model: {a.source_representation}/{a.original_feature_count}; ANOVA-{a.selected_features}; {a.scaler_type}; RBF SVC C={dict(a.svm_parameters)['C']:g} gamma={dict(a.svm_parameters)['gamma']}; probability=False")
    print(f"svm_top1 global threshold={context.threshold:.12g}; window=7 median; identity majority vote; consistency gate=disabled")
    for role, count in (("KNOWN", result.known_final), ("UNKNOWN", result.unknown_final)):
        print(f"Final {role}: identities={count.identities}; sessions={count.sessions}; samples={count.samples}")
    print(f"FINAL CLOSED-SET FRAMES: accuracy={closed.metrics.accuracy:.6f}; macro precision={closed.metrics.macro_precision:.6f}; macro recall={closed.metrics.macro_recall:.6f}; macro F1={closed.metrics.macro_f1:.6f}")
    closed_delta = dict(result.closed_set_delta_final_minus_development)
    print(f"Descriptive closed-set deltas vs development: accuracy={closed_delta['accuracy']:+.6f}; macro F1={closed_delta['macro_f1']:+.6f}")
    print(f"FINAL TEMPORAL: known windows={result.known_windows.windows}; unknown windows={result.unknown_windows.windows}; known discarded={result.known_windows.discarded_samples}; unknown discarded={result.unknown_windows.discarded_samples}")
    print(f"Diagnostic ROC AUC={result.temporal_score_diagnostics.roc_auc:.6f}; EER={result.temporal_score_diagnostics.equal_error_point.eer:.6f}; EER threshold is NEVER operational")
    print(f"Frozen-threshold FAR={m.unknown_false_acceptance_rate:.6f}; FRR={m.known_rejection_rate:.6f}; known acceptance={m.known_acceptance_rate:.6f}; unknown rejection={m.unknown_rejection_rate:.6f}")
    print(f"Correct known ID={m.known_correct_identification_rate:.6f}; accepted misidentification={m.known_misidentification_rate:.6f}; open accuracy={m.open_set_exact_accuracy:.6f}; balanced accuracy={m.binary_balanced_accuracy:.6f}; binary macro F1={m.binary_macro_f1:.6f}")
    print(f"Development selection (NOT final performance): AUC={result.development_candidate.roc_auc:.6f}; open accuracy={result.development_candidate.open_set_accuracy:.6f}; descriptive final deltas={result.delta_final_minus_development.roc_auc:+.6f}/{result.delta_final_minus_development.open_set_accuracy:+.6f}")
    print(f"OOF calibration (NOT final performance): AUC={context.calibration_roc_auc:.6f}; balanced accuracy={context.calibration_balanced_accuracy:.6f}")
    print("Independent final KNOWN/UNKNOWN biometric payloads accessed: yes. Model artifacts unchanged: yes.")
    print(V2_NO_RETUNING)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
