"""One-time final holdout evaluation: no fitting, calibration or parameter overrides."""

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from face_recognition.application.evaluate_final_open_set import FinalOpenSetEvaluator
from face_recognition.application.final_evaluation_data import FinalEvaluationError
from face_recognition.domain.models.baseline_evaluation import EvaluationVersions
from face_recognition.domain.models.final_evaluation import CALIBRATION_WARNING, NO_RETUNING, FinalModelContext
from face_recognition.infrastructure.classifiers.sklearn_final_open_set import SklearnFinalOpenSetRunner
from face_recognition.infrastructure.persistence.baseline_evaluation_report import validate_aggregate_report_target
from face_recognition.infrastructure.persistence.final_evaluation_report import write_final_evaluation_report
from face_recognition.infrastructure.persistence.final_holdout_loader import FinalHoldoutLoader, validate_final_provenance
from face_recognition.infrastructure.persistence.identity_split_repository import read_identity_split
from face_recognition.infrastructure.persistence.model_repository import load_open_set_model, model_fingerprints
from face_recognition.infrastructure.persistence.open_set_config_repository import read_frozen_configuration, read_aggregate_development_report
from face_recognition.infrastructure.persistence.unknown_rejection_repository import read_unknown_split


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Final holdout evaluation only. Do not use these results to retune this model version.")
    for name in ("processed-root", "identity-split", "unknown-split", "model-dir", "frozen-config", "development-report", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        inputs = tuple(p.resolve() for p in (args.processed_root, args.identity_split, args.unknown_split, args.model_dir,
                                            args.frozen_config, args.development_report))
        output = validate_aggregate_report_target(args.report, inputs)  # Refuse overwrite BEFORE any holdout access.
        before = model_fingerprints(args.model_dir)
        frozen = read_frozen_configuration(args.frozen_config)
        model = load_open_set_model(args.model_dir, frozen)
        source = read_identity_split(args.identity_split)
        partition = read_unknown_split(args.unknown_split, source)
        development = read_aggregate_development_report(args.development_report)
        validate_final_provenance(model, frozen, source, partition, development)
        if model_fingerprints(args.model_dir) != before:
            raise FinalEvaluationError("Model changed before holdout access")
        context = FinalModelContext(frozen.configuration, frozen.sha256, before, model.threshold,
                                    model.manifest["calibration_analysis"]["roc_auc"],
                                    model.manifest["calibration_analysis"]["calibration"]["metrics"]["binary_balanced_accuracy"],
                                    EvaluationVersions(**model.manifest["versions"]))
        data = FinalHoldoutLoader(args.processed_root).load_final(source, partition, model.parts.class_ids,
                                                                frozen.configuration.architecture.original_feature_count)
        result = FinalOpenSetEvaluator(SklearnFinalOpenSetRunner(model)).evaluate(data, context)
        if model_fingerprints(args.model_dir) != before:
            raise FinalEvaluationError("Model changed during final evaluation")
        write_final_evaluation_report(result, output, inputs, args.model_dir)
    except KeyboardInterrupt:
        print("Final evaluation interrupted; check whether a completed report exists before retrying. Do not retune this model version.", file=sys.stderr)
        return 1
    except Exception:
        print("Final evaluation failed: invalid input/provenance, changed model, existing report or inconsistent final data. No retuning was performed.", file=sys.stderr)
        return 1
    a, m, closed = frozen.configuration.architecture, result.operational_metrics, result.closed_set_known_frames
    parameters = dict(a.svm_parameters)
    print("FINAL OPEN-SET HOLDOUT EVALUATION")
    print(f"Model: ANOVA-{a.selected_features} / {a.scaler_type} / {parameters['kernel']} SVC "
          f"C={parameters['C']:g} gamma={parameters['gamma']:g} probability={parameters['probability']}; "
          f"{a.novelty_method}; window={a.temporal_window_size}; threshold={context.threshold:.12g}")
    print("\nCALIBRATION PERFORMANCE (NOT GENERALIZATION)")
    print(f"Calibration AUC={context.calibration_roc_auc:.6f}; balanced accuracy={context.calibration_balanced_accuracy:.6f}")
    print(CALIBRATION_WARNING)
    print("\nCLOSED-SET KNOWN TEST — FINAL HELD-OUT SESSION")
    print(f"Samples={closed.samples}; accuracy={closed.metrics.accuracy:.6f}; macro precision={closed.metrics.macro_precision:.6f}; "
          f"macro recall={closed.metrics.macro_recall:.6f}; macro F1={closed.metrics.macro_f1:.6f}")
    print("\nFINAL OPEN-SET TEMPORAL TEST")
    print(f"Known windows={result.known_windows.windows}; unknown windows={result.unknown_windows.windows}")
    print(f"Known used/discarded={result.known_windows.used_samples}/{result.known_windows.discarded_samples}; "
          f"unknown used/discarded={result.unknown_windows.used_samples}/{result.unknown_windows.discarded_samples}")
    print(f"ROC AUC={result.temporal_novelty_diagnostics.roc_auc:.6f}; EER={result.temporal_novelty_diagnostics.equal_error_point.eer:.6f} (diagnostics only)")
    print(f"Frozen-threshold FAR={m.unknown_false_acceptance_rate:.6f}; FRR={m.known_rejection_rate:.6f}; "
          f"known acceptance={m.known_acceptance_rate:.6f}; unknown rejection={m.unknown_rejection_rate:.6f}")
    print(f"Known correct ID={m.known_correct_identification_rate:.6f}; accepted-known misidentification={m.known_misidentification_rate:.6f}; "
          f"open-set accuracy={m.open_set_exact_accuracy:.6f}; balanced accuracy={m.binary_balanced_accuracy:.6f}; binary macro F1={m.binary_macro_f1:.6f}")
    print(f"Development comparison (descriptive only): AUC={result.development_candidate.roc_auc:.6f}, "
          f"open-set accuracy={result.development_candidate.open_set_accuracy:.6f}; "
          f"final deltas={result.delta_final_minus_development.roc_auc:+.6f}/{result.delta_final_minus_development.open_set_accuracy:+.6f}")
    print("known_test biometric payloads accessed: yes")
    print("unknown_holdout biometric payloads accessed: yes")
    print("Model artifacts unchanged: yes. FINAL HOLDOUT EVALUATION COMPLETE")
    print(NO_RETUNING)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
