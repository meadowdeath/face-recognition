"""External dataset processing CLI: 0 success, 1 failure, argparse 2 usage error."""

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from face_recognition.application.process_dataset import DatasetProcessor, DatasetProcessingError
from face_recognition.domain.models.processed_sample import (
    BASELINE_REPRESENTATION, NORMALIZED_REPRESENTATION, REPRESENTATIONS,
)
from face_recognition.infrastructure.features.landmark_feature_extractor import LandmarkFeatureExtractor
from face_recognition.infrastructure.features.landmark_normalizer import LandmarkNormalizer
from face_recognition.infrastructure.persistence.processed_dataset_repository import (
    FilesystemProcessedDatasetRepository, ProcessedDatasetPersistenceError,
)
from face_recognition.infrastructure.persistence.raw_dataset_reader import RawDatasetReader, RawDatasetReadError


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Process external raw landmarks into normalized and all-XYZ datasets")
    parser.add_argument("--raw-root", required=True, type=Path, help="Existing external raw dataset directory")
    parser.add_argument("--output-root", required=True, type=Path, help="New external processed directory; must not exist")
    args = parser.parse_args(argv)

    try:
        # The adapters resolve paths and enforce safety. Pass the original output
        # Path so resolving it here cannot bypass the repository's symlink check.
        reader = RawDatasetReader(args.raw_root)
        repository = FilesystemProcessedDatasetRepository(args.output_root, reader.root)
        processor = DatasetProcessor(LandmarkNormalizer, LandmarkFeatureExtractor())
        summary = processor.process(reader, repository)
    except (RawDatasetReadError, ProcessedDatasetPersistenceError, DatasetProcessingError) as exc:
        print(f"Processing failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Processing interrupted.", file=sys.stderr)
        return 1
    except Exception:
        # Unexpected exceptions are not guaranteed to have sanitized messages.
        print("Processing failed: unexpected internal or filesystem error.", file=sys.stderr)
        return 1

    print("Processing completed.")
    print(f"Participants: {summary.participants}")
    print(f"Sessions: {summary.sessions}")
    print(f"Conditions/files per representation: {summary.conditions}")
    print(f"Total condition files: {summary.conditions * len(REPRESENTATIONS)}")
    print(f"Samples: {summary.total_samples_processed}")
    print(f"Rejected samples: {summary.rejected_samples}")
    print(f"Errors: {summary.error_count}")
    print(f"Normalized representation: {NORMALIZED_REPRESENTATION}")
    print(f"Feature representation: {BASELINE_REPRESENTATION}")
    print(f"Output directory: {repository.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
