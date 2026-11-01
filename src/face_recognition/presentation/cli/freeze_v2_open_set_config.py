"""Freeze V2 from aggregate evidence only: exit 0 success, 1 failure, 2 usage."""

import argparse
from pathlib import Path
import sys

from face_recognition.application.freeze_v2_open_set_config import freeze_v2_configuration
from face_recognition.domain.models.open_set_config import ConfigurationFreezeError
from face_recognition.infrastructure.persistence.baseline_evaluation_report import BaselineReportError, validate_aggregate_report_target
from face_recognition.infrastructure.persistence.open_set_config_repository import read_aggregate_development_report
from face_recognition.infrastructure.persistence.v2_open_set_config_repository import write_frozen_v2_configuration


def main(argv=None):
    parser = argparse.ArgumentParser(description="Freeze V2 architecture and OOF threshold policy BEFORE new final-test collection; aggregate JSON only")
    parser.add_argument("--diagnostic-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        source = args.diagnostic_report.resolve()
        target = validate_aggregate_report_target(args.output, (source,))
        report = read_aggregate_development_report(source)
        config = freeze_v2_configuration(report.payload, report.sha256)
        write_frozen_v2_configuration(config, target, source)
    except (ConfigurationFreezeError, BaselineReportError) as error:
        print(f"V2 configuration freeze failed: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("V2 configuration freeze interrupted.", file=sys.stderr)
        return 1
    except Exception:
        print("V2 configuration freeze failed: internal or filesystem error.", file=sys.stderr)
        return 1
    print("V2 OPEN-SET DEVELOPMENT CONFIGURATION FROZEN")
    print("Identity: all_xyz_v1 (1434) / ANOVA-448 / StandardScaler / RBF SVM C=4 gamma=0.001; probability=False")
    print("Rejection: svm_top1; one global threshold; higher score means KNOWN")
    print("Temporal: window=7; median score; majority-vote identity; consistency gate=disabled")
    print("Final numeric threshold: not yet produced; cross-session OOF policy frozen=yes")
    print("Threshold transfer: direct to later all-session refit; development-only scale stability checks required")
    print("Future independent test: new known session required=yes; new UNKNOWN identities required=yes")
    print("V2 final evaluation performed=no; final model fitted=no; biometric payloads accessed=no")
    print("Exact aggregate report SHA-256 recorded; V1 remains immutable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
