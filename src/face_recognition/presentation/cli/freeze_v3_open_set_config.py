"""Freeze V3 decisions from aggregate evidence, without model/data access."""

import argparse
from pathlib import Path
import sys

from face_recognition.domain.models.v3_open_set_config import EVIDENCE_KINDS
from face_recognition.infrastructure.persistence.v3_open_set_config import freeze_v3_configuration_file


def main(argv=None):
    parser = argparse.ArgumentParser(description="Freeze selected V3 configuration and evidence; no fitting or calibration")
    for kind in EVIDENCE_KINDS:
        parser.add_argument("--" + kind.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        paths = {kind: getattr(args, kind) for kind in EVIDENCE_KINDS}
        config = freeze_v3_configuration_file(paths, args.output)
    except KeyboardInterrupt:
        print("V3 freeze interrupted; check for a completed configuration before retrying.", file=sys.stderr)
        return 1
    except Exception:
        print("V3 freeze failed: invalid/incompatible evidence, provenance or existing output. No partial configuration published.", file=sys.stderr)
        return 1
    print("V3 OPEN-SET DEVELOPMENT CONFIGURATION FROZEN")
    print(f"Representation: {config.pipeline.representation}; input features={config.pipeline.input_features}")
    print("ANOVA f_classif k=448 / StandardScaler / RBF SVC C=4 gamma=0.001 probability=False")
    print("Novelty: regularized predicted-class Mahalanobis; score=-distance; higher means KNOWN")
    print("Temporal: non-overlapping window=7 / median / deterministic plurality and score-sum/class-order ties / gate disabled")
    print("Threshold transport: none_same_model_calibration; no post-calibration refit")
    print("Numeric threshold: not calibrated; deployable-session allocation: not defined")
    print("Evidence: five SHA-256 fingerprints stored; no biometric payloads or estimators accessed")
    print("V3 is iterative cross-validation development, not independent final generalization. V2 remains the last independent external holdout.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
