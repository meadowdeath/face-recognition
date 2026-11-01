"""Create/validate independent final role metadata, without opening any JSONL."""

import argparse
from pathlib import Path
import sys

from face_recognition.infrastructure.persistence.baseline_evaluation_report import validate_aggregate_report_target
from face_recognition.infrastructure.persistence.final_v2_holdout_repository import (
    FinalV2HoldoutLoader, read_final_v2_manifest, read_v2_final_inputs, write_final_v2_manifest,
)
from face_recognition.infrastructure.persistence.v2_model_repository import v2_model_fingerprints


def main(argv=None):
    parser = argparse.ArgumentParser(description="Catalog-only final V2 role declaration/validation; no fitting, inference or feature inspection")
    for name in ("processed-root", "identity-split", "unknown-split", "model-dir", "frozen-config", "development-report", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true", help="Validate existing --output without changing it or reading feature files")
    args = parser.parse_args(argv)
    try:
        paths = tuple(p.resolve() for p in (args.processed_root, args.identity_split, args.unknown_split,
                      args.model_dir, args.frozen_config, args.development_report))
        if not args.validate_only:
            validate_aggregate_report_target(args.output, paths)
        inputs = read_v2_final_inputs(args.identity_split, args.unknown_split, args.frozen_config, args.development_report, args.model_dir)
        loader = FinalV2HoldoutLoader(args.processed_root)
        if args.validate_only:
            manifest, _ = read_final_v2_manifest(args.output)
            loader.authorize(manifest, inputs)
        else:
            manifest = loader.catalog_manifest(inputs)
        if v2_model_fingerprints(args.model_dir) != inputs.model_fingerprints:
            raise ValueError
        if not args.validate_only:
            write_final_v2_manifest(manifest, args.output, paths)
    except KeyboardInterrupt:
        print("Final V2 catalog authorization interrupted.", file=sys.stderr)
        return 1
    except Exception:
        print("Final V2 manifest failed: invalid provenance, reused/dependent roles, incomplete or mismatched catalog, changed model, or existing output.", file=sys.stderr)
        return 1
    print("FINAL V2 HOLDOUT MANIFEST " + ("VALIDATED" if args.validate_only else "CREATED"))
    print(f"KNOWN identities/sessions={len({s.person_id for s in manifest.known_sessions})}/{len(manifest.known_sessions)}; UNKNOWN identities/sessions={len({s.person_id for s in manifest.unknown_sessions})}/{len(manifest.unknown_sessions)}")
    print(f"Condition files={len(manifest.condition_files)}; total samples declared by processing manifest={manifest.expected_total_samples}")
    print("Catalog/metadata only: no JSONL values inspected, no estimator deserialized, no fitting or inference. Record validation occurs before final inference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
