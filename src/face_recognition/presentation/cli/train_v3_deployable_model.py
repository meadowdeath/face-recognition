"""Construct one frozen V3 deployable bundle; calibration is not evaluation."""

import argparse
from pathlib import Path
import sys

from face_recognition.application.train_v3_deployable_model import TrainV3DeployableModel
from face_recognition.infrastructure.classifiers.sklearn_v3_deployable import SklearnV3DeployableRunner
from face_recognition.infrastructure.persistence.aggregate_report import validate_aggregate_report_target
from face_recognition.infrastructure.persistence.v3_deployable_dataset import V3DeployableInputs
from face_recognition.infrastructure.persistence.v3_deployable_model import publish_v3_deployable_bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train frozen V3 once and calibrate on separate sessions with the same fitted state")
    for name in ("processed-root", "frozen-config", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--protocol", type=Path, help="Optional relocated protocol; must match frozen SHA-256")
    parser.add_argument("--pose-report", type=Path, help="Optional relocated pose report; must match frozen SHA-256")
    args = parser.parse_args(argv)
    try:
        early_inputs = tuple(p for p in (args.processed_root, args.frozen_config, args.protocol, args.pose_report) if p is not None)
        validate_aggregate_report_target(args.output_dir, early_inputs)
        inputs = V3DeployableInputs(args.processed_root, args.frozen_config, args.protocol, args.pose_report)
        validate_aggregate_report_target(args.output_dir, inputs.input_paths)
        print("FROZEN V3 DEPLOYABLE MODEL CONSTRUCTION — no selection or evaluation", flush=True)
        data, provenance = inputs.load()
        runner = SklearnV3DeployableRunner(inputs.config)
        result = TrainV3DeployableModel(runner, progress=lambda text: print(text, flush=True)).train(data, inputs.config, provenance)
        inputs.verify()
        publish_v3_deployable_bundle(result, inputs.config_path, args.output_dir, inputs.input_paths, inputs.verify)
    except KeyboardInterrupt:
        print("Training interrupted; check for a completed bundle before retrying.", file=sys.stderr)
        return 1
    except Exception:
        print("V3 construction failed: invalid data/configuration, changed fitted state or unsafe/existing output. No partial completed bundle published.", file=sys.stderr)
        return 1
    summary = result.summary
    print(f"KNOWN fit samples={summary.known_fit_samples}; classes={len(summary.allocation.known_identities)}; selected features=448; support vectors={summary.support_vectors}")
    print(f"KNOWN calibration frames={summary.known_calibration.source_samples}; windows={summary.known_calibration.windows}; discarded={summary.known_calibration.discarded_samples}")
    print(f"UNKNOWN calibration frames={summary.unknown_calibration.source_samples}; windows={summary.unknown_calibration.windows}; discarded={summary.unknown_calibration.discarded_samples}")
    print(f"Threshold={summary.threshold:.12g}; threshold_transport={summary.threshold_transport}; fit generation=1")
    m, d = summary.calibration_metrics, summary.calibration_diagnostics
    print("CALIBRATION METRICS — not independent final test/generalization performance")
    print(f"FAR={m.unknown_false_acceptance_rate:.6f}; FRR={m.known_rejection_rate:.6f}; balanced accuracy={m.binary_balanced_accuracy:.6f}; binary macro F1={m.binary_macro_f1:.6f}")
    print(f"KNOWN acceptance={m.known_acceptance_rate:.6f}; UNKNOWN rejection={m.unknown_rejection_rate:.6f}; diagnostic AUC={d.roc_auc:.6f}; EER={d.equal_error_point.eer:.6f}")
    print("State fingerprints unchanged. Local biometric-derived bundle persisted and integrity-validated.")
    print("No new V3 evaluation performed. V2 remains the last independent external holdout.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
