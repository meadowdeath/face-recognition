"""Aggregate-only diagnostics for session-aware development folds; no fitting."""

import argparse
from pathlib import Path
import sys

from face_recognition.application.training_folds import TrainingFoldError, create_session_folds
from face_recognition.application.validate_identity_split import SplitValidationError
from face_recognition.infrastructure.persistence.development_dataset_loader import DevelopmentDatasetLoader
from face_recognition.infrastructure.persistence.identity_split_repository import IdentitySplitPersistenceError, read_identity_split
from face_recognition.infrastructure.persistence.processed_dataset_catalog import ProcessedDatasetReadError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = read_identity_split(args.split_manifest.resolve())
        data = DevelopmentDatasetLoader(args.processed_root.resolve()).load(manifest)
        folds = create_session_folds(data)
        for fold in folds:
            print(f"{fold.name}:")
            print(f"  Fit identities: {fold.fit.identity_count}")
            print(f"  Fit samples: {len(fold.fit.indices)}")
            print(f"  Validation identities: {fold.validation.identity_count}")
            print(f"  Validation samples: {len(fold.validation.indices)}")
        return 0
    except (TrainingFoldError, SplitValidationError, IdentitySplitPersistenceError, ProcessedDatasetReadError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except Exception:
        print("Error: development-fold inspection failed", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
