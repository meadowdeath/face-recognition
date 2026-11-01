"""Explicit identity/session split CLI: 0 success, 1 failure, 2 usage error."""

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from face_recognition.application.create_identity_split import IdentitySplitError, IdentitySplitPlanner
from face_recognition.domain.models.dataset import validate_identifier
from face_recognition.domain.models.identity_split import IdentityPartition, IdentitySplitSpec
from face_recognition.infrastructure.persistence.feature_metadata_reader import FeatureMetadataReader, FeatureMetadataReadError
from face_recognition.infrastructure.persistence.identity_split_repository import IdentitySplitPersistenceError, write_identity_split


def _identifier(value: str) -> str:
    try:
        return validate_identifier(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a logical all-XYZ identity split using whole sessions")
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--known-id", action="append", type=_identifier, default=[])
    parser.add_argument("--unknown-id", action="append", type=_identifier, default=[])
    parser.add_argument("--train-session", action="append", type=_identifier, required=True)
    parser.add_argument("--known-test-session", action="append", type=_identifier, required=True)
    args = parser.parse_args(argv)
    try:
        specification = IdentitySplitSpec(tuple(args.known_id), tuple(args.unknown_id),
                                           tuple(args.train_session), tuple(args.known_test_session))
    except ValueError as exc:
        print(f"Split failed: {exc}", file=sys.stderr)
        return 1
    try:
        source = FeatureMetadataReader(args.processed_root)
        manifest = IdentitySplitPlanner().plan(source, specification)
        write_identity_split(manifest, args.output, source.processed_root)
    except (IdentitySplitError, FeatureMetadataReadError, IdentitySplitPersistenceError) as exc:
        print(f"Split failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Split creation interrupted.", file=sys.stderr)
        return 1
    except Exception:
        print("Split failed: invalid source/output or internal/filesystem error.", file=sys.stderr)
        return 1
    counts = {count.partition: count.samples for count in manifest.partition_counts}
    print("Identity split created.")
    print(f"Known identities: {manifest.known_identity_count}")
    print(f"Unknown identities: {manifest.unknown_identity_count}")
    print(f"Known training samples: {counts[IdentityPartition.KNOWN_TRAIN_POOL]}")
    print(f"Known test samples: {counts[IdentityPartition.KNOWN_TEST]}")
    print(f"Unknown pool samples: {counts[IdentityPartition.UNKNOWN_POOL]}")
    print(f"Total samples: {manifest.total_sample_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
