"""Validate a dynamic pose population and construct one calibrated bundle."""

import argparse
from pathlib import Path
import sys

from face_recognition.application.train_recognition_model import TrainRecognitionModel
from face_recognition.domain.models.dataset import CONDITIONS
from face_recognition.domain.models.rejection import RejectionError
from face_recognition.infrastructure.classifiers.sklearn_recognition import RecognitionModelRunner
from face_recognition.infrastructure.persistence.artifact_integrity import ArtifactPublicationError, validate_output_target
from face_recognition.infrastructure.persistence.dynamic_training_dataset import DynamicTrainingInputs
from face_recognition.infrastructure.persistence.recognition_model import load_recognition_bundle, publish_dynamic_bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path)
    parser.add_argument("--population-config", type=Path)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--output-dir", type=Path, help="Fresh external model bundle directory")
    actions.add_argument("--validate-only", action="store_true", help="Validate captures and provenance without fitting")
    actions.add_argument("--verify-bundle", type=Path, help="Verify an existing trusted local recognition bundle")
    args = parser.parse_args(argv)
    if args.verify_bundle is not None:
        if args.processed_root or args.population_config:
            parser.error("--verify-bundle only needs the bundle path")
    elif args.processed_root is None or args.population_config is None:
        parser.error("Training and preflight require --processed-root and --population-config")
    try:
        if args.verify_bundle is not None:
            load_recognition_bundle(args.verify_bundle)
            print("Recognition bundle and runtime compatibility verified")
            return 0
        if args.output_dir is not None:
            validate_output_target(args.output_dir, (args.processed_root, args.population_config))
        inputs = DynamicTrainingInputs(args.processed_root, args.population_config)
        p = inputs.population
        print(f"Population valid: {len(p.known_ids)} KNOWN, {len(p.unknown_ids)} UNKNOWN "
              f"(minimum UNKNOWN: {p.minimum_unknown})", flush=True)
        print("Validating complete capture groups and feature payloads", flush=True)
        data, config, allocation = inputs.load()
        per_session = allocation.samples_per_condition * len(CONDITIONS)
        print(f"Complete sessions: {len(p.known_ids)*4} KNOWN and {len(p.unknown_ids)*2} UNKNOWN")
        print(f"Samples: {len(p.known_ids)*3*per_session} fit, {len(p.known_ids)*per_session} KNOWN calibration, "
              f"{len(p.unknown_ids)*2*per_session} UNKNOWN calibration", flush=True)
        if args.validate_only:
            print("Capture protocol, source digests and model configuration verified. Training may proceed.")
            return 0
        runner = RecognitionModelRunner(config)
        result = TrainRecognitionModel(runner, progress=lambda message: print(message, flush=True)).train_dynamic(data, config)
        publish_dynamic_bundle(result, args.output_dir, inputs.input_paths, inputs.verify)
    except KeyboardInterrupt:
        print("Dynamic construction interrupted. Check for a completed bundle before retrying.", file=sys.stderr)
        return 1
    except (ArtifactPublicationError, RejectionError) as exc:
        print(f"Dynamic construction failed: {exc}", file=sys.stderr)
        return 1
    except Exception:
        print("Dynamic construction failed: invalid source, runtime or unsafe output.", file=sys.stderr)
        return 1
    summary = result.summary
    print(f"Published complete dynamic bundle. Threshold={summary.threshold:.12g}")
    print("CALIBRATION ONLY: "
          f"balanced accuracy={summary.calibration_metrics.binary_balanced_accuracy:.6f}, "
          f"UNKNOWN FAR={summary.calibration_metrics.unknown_false_acceptance_rate:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
