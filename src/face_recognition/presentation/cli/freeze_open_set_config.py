"""Freeze aggregate development choices: 0 success, 1 failure, 2 usage error."""

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from face_recognition.application.freeze_open_set_config import freeze_development_configuration
from face_recognition.domain.models.open_set_config import ConfigurationFreezeError
from face_recognition.infrastructure.persistence.baseline_evaluation_report import BaselineReportError, validate_aggregate_report_target
from face_recognition.infrastructure.persistence.open_set_config_repository import read_aggregate_development_report, write_frozen_configuration


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze fixed open-set choices from an aggregate development report; never fit or evaluate holdouts")
    parser.add_argument("--development-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        source = args.development_report.resolve()
        target = validate_aggregate_report_target(args.output, (source,))
        document = read_aggregate_development_report(source)
        configuration = freeze_development_configuration(document.payload, document.sha256)
        write_frozen_configuration(configuration, target, source)
    except (ConfigurationFreezeError, BaselineReportError) as error:
        print(f"Configuration freeze failed: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Configuration freeze interrupted.", file=sys.stderr)
        return 1
    except Exception:
        print("Configuration freeze failed: internal or filesystem error.", file=sys.stderr)
        return 1
    print("OPEN-SET DEVELOPMENT CONFIGURATION FROZEN")
    print("Representation: all_xyz_v1 (1434 original features)")
    print("Features: ANOVA k=448; StandardScaler")
    print("Classifier: RBF SVM C=4 gamma=0.001; probability=False")
    print("Novelty: mean_5nn / Euclidean; higher negative-distance score means KNOWN")
    print("Temporal window: 7; novelty aggregation: median; identity aggregation: majority vote")
    print("Final threshold: not numerically calibrated yet; policy frozen: yes")
    print("known_test accessed: no")
    print("unknown_holdout accessed: no")
    print("Aggregate source report SHA-256 recorded; no model fitted or holdout evaluated.")
    print("Configuration frozen successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
