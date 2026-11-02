"""Create/validate metadata-only V3 CV protocol: 0 success, 1 failure, 2 usage."""

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from face_recognition.application.create_v3_cv_protocol import create_v3_cv_protocol
from face_recognition.domain.models.v3_cv_protocol import V3ProtocolError
from face_recognition.infrastructure.persistence.aggregate_report import validate_aggregate_report_target
from face_recognition.infrastructure.persistence.v3_cv_protocol_repository import (
    read_v3_catalog, read_v3_cv_protocol, read_v3_experiment_specification, write_v3_cv_protocol,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Metadata-only nested session/identity CV; no fitting, scoring or selection")
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--experiment-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true", help="Validate an existing protocol against current metadata; write nothing")
    args = parser.parse_args(argv)
    try:
        inputs = (args.processed_root, args.experiment_spec)
        if not args.validate_only:
            validate_aggregate_report_target(args.output, inputs)  # Refuse overwrite before source inspection.
        spec = read_v3_experiment_specification(args.experiment_spec)
        catalog = read_v3_catalog(args.processed_root)
        protocol = create_v3_cv_protocol(spec, catalog)
        if args.validate_only:
            if read_v3_cv_protocol(args.output) != protocol:
                raise V3ProtocolError("Catalog/specification changed since protocol creation")
        else:
            write_v3_cv_protocol(protocol, args.output, inputs)
    except KeyboardInterrupt:
        print("V3 protocol operation interrupted.", file=sys.stderr)
        return 1
    except Exception:
        print("V3 protocol failed: invalid/incomplete catalog, assignments, digests or existing output.", file=sys.stderr)
        return 1
    print("V3 NESTED CROSS-VALIDATION PROTOCOL " + ("VALIDATED" if args.validate_only else "CREATED"))
    print(f"Representation: {catalog.representation}")
    print(f"KNOWN identities: {len(spec.known_ids)}; UNKNOWN identities: {len(spec.unknown_ids)}")
    print(f"Catalog sessions: {len(spec.known_ids) * 4 + len(spec.unknown_ids) * 2}; condition files: {len(catalog.condition_files)}")
    print(f"Declared source samples (not payload-validated): {catalog.declared_source_samples}")
    print("Outer folds: 4; UNKNOWN test identities per outer fold: 2")
    print("Inner splits per outer fold: 18; KNOWN fit/calibration/validation sessions: 1/1/1")
    print("Inner UNKNOWN calibration/validation identities: 4/2; no UNKNOWN fitting")
    print(f"Protocol SHA-256: {protocol.protocol_sha256}")
    print("Biometric payloads accessed: no. Fitting/scoring/threshold search/model selection performed: no.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
