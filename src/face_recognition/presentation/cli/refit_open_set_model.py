"""Frozen development refit/calibration: 0 success, 1 failure, 2 usage."""

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from face_recognition.application.refit_open_set_model import OpenSetModelRefitter
from face_recognition.infrastructure.classifiers.sklearn_open_set_refit import SklearnOpenSetRefitRunner
from face_recognition.infrastructure.persistence.development_dataset_loader import DevelopmentDatasetLoader
from face_recognition.infrastructure.persistence.identity_split_repository import read_identity_split
from face_recognition.infrastructure.persistence.model_repository import load_open_set_model, persist_open_set_model, validate_model_target
from face_recognition.infrastructure.persistence.open_set_config_repository import read_frozen_configuration
from face_recognition.infrastructure.persistence.unknown_calibration_loader import UnknownCalibrationLoader
from face_recognition.infrastructure.persistence.unknown_rejection_repository import read_unknown_split


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Refit/calibrate frozen model using development data only; no holdout evaluation")
    for argument in ("processed-root", "identity-split", "unknown-split", "frozen-config", "output-dir"):
        parser.add_argument("--" + argument, required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        inputs = tuple(path.resolve() for path in (args.processed_root, args.identity_split, args.unknown_split, args.frozen_config))
        target = validate_model_target(args.output_dir, inputs)
        frozen = read_frozen_configuration(args.frozen_config)
        source = read_identity_split(args.identity_split)
        partition = read_unknown_split(args.unknown_split, source)
        known = DevelopmentDatasetLoader(args.processed_root).load(source)
        unknown = UnknownCalibrationLoader(args.processed_root).load_calibration(source, partition, known.X.shape[1])
        result = OpenSetModelRefitter(SklearnOpenSetRefitRunner()).run(frozen.configuration, frozen.sha256, known, unknown)
        summary = result.summary
        del known, unknown
        persist_open_set_model(result, frozen, target, inputs)
        del result
        loaded = load_open_set_model(target, frozen)
        del loaded
    except KeyboardInterrupt:
        print("Frozen model refit interrupted; no holdout evaluation performed.", file=sys.stderr)
        return 1
    except Exception:
        print("Frozen model refit failed: invalid configuration/data, unsafe or existing output, or artifact-integrity error.", file=sys.stderr)
        return 1
    analysis, metrics = summary.analysis, summary.analysis.calibration.metrics
    architecture = frozen.configuration.architecture
    svm_parameters = dict(architecture.svm_parameters)
    print("FROZEN OPEN-SET MODEL REFIT AND CALIBRATION")
    print(f"Frozen configuration: ANOVA-{architecture.selected_features} / {architecture.scaler_type} / "
          f"{svm_parameters['kernel']} {architecture.classifier_type} C={svm_parameters['C']:g} gamma={svm_parameters['gamma']:g} / "
          f"{architecture.novelty_method} / window={architecture.temporal_window_size} {architecture.novelty_temporal_aggregation}")
    print(f"Known training samples={summary.known_training_samples}; identities={summary.identity_count}; "
          f"selected features={summary.selected_feature_count}; support vectors={summary.support_vector_count}")
    print(f"Known calibration windows={summary.known_calibration.windows}; unknown calibration windows={summary.unknown_calibration.windows}")
    print(f"Discarded known samples={summary.known_calibration.discarded_samples}; unknown={summary.unknown_calibration.discarded_samples}")
    print(f"Threshold={analysis.calibration.threshold:.12g}; ROC AUC={analysis.roc_auc:.6f}; EER={analysis.equal_error_point.eer:.6f}")
    print(f"FAR={metrics.unknown_false_acceptance_rate:.6f}; FRR={metrics.known_rejection_rate:.6f}; "
          f"balanced accuracy={metrics.binary_balanced_accuracy:.6f}")
    for point in analysis.far_operating_points:
        if point.metrics:
            print(f"Diagnostic FAR<={100 * point.target_far:g}%: {point.status}; FAR={point.metrics.unknown_false_acceptance_rate:.6f}; "
                  f"FRR={point.metrics.known_rejection_rate:.6f}; threshold={point.threshold:.12g}")
    print("Calibration is not generalization accuracy; known queries use training data with explicit self-exclusion.")
    print("known_test biometric payloads accessed: no")
    print("unknown_holdout biometric payloads accessed: no")
    print("Model artifacts persisted locally and integrity validated. Final holdout evaluation has NOT been performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
