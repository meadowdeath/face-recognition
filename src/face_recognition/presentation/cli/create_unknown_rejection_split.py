"""Create or verify identity-level unknown roles using metadata only."""

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from face_recognition.application.unknown_rejection_split import plan_unknown_split
from face_recognition.domain.models.unknown_rejection import UnknownSplitSpec
from face_recognition.infrastructure.persistence.baseline_evaluation_report import validate_aggregate_report_target
from face_recognition.infrastructure.persistence.identity_split_repository import read_identity_split
from face_recognition.infrastructure.persistence.unknown_rejection_repository import read_unknown_split, write_unknown_split


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-split", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path, help="Validate an existing split and print counts without writing")
    parser.add_argument("--calibration-id", action="append")
    parser.add_argument("--holdout-id", action="append")
    args = parser.parse_args(argv)
    if args.verify is not None and (args.calibration_id or args.holdout_id):
        parser.error("Verification reads roles from the manifest; do not supply role arguments")
    if args.output is not None and (not args.calibration_id or not args.holdout_id):
        parser.error("Creation requires explicit --calibration-id and --holdout-id assignments")
    try:
        source_path = args.identity_split.resolve()
        source = read_identity_split(source_path)
        if args.verify is not None:
            split = read_unknown_split(args.verify.resolve(), source)
        else:
            target = validate_aggregate_report_target(args.output, (source_path,))
            split = plan_unknown_split(source, UnknownSplitSpec(tuple(args.calibration_id), tuple(args.holdout_id)))
            write_unknown_split(split, source, target, source_path)
    except ValueError as error:
        print(f"Unknown split failed: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Unknown split interrupted.", file=sys.stderr)
        return 1
    except Exception:
        print("Unknown split failed: internal or filesystem error.", file=sys.stderr)
        return 1
    print("Unknown rejection split verified." if args.verify is not None else "Unknown rejection split created.")
    for name, counts in (("Unknown calibration", split.calibration), ("Unknown holdout", split.holdout)):
        print(f"{name}: identities={counts.identities}; sessions={counts.sessions}; "
              f"condition files={counts.condition_files}; samples={counts.samples}")
    print("Biometric feature payloads accessed: no")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
