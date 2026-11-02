"""Process raw capture into a fresh, pose-only external representation."""

import argparse
from pathlib import Path
import sys

from face_recognition.application.process_dataset import DatasetProcessor, DatasetProcessingError
from face_recognition.domain.models.processed_sample import POSE_REPRESENTATION
from face_recognition.infrastructure.features.landmark_feature_extractor import LandmarkFeatureExtractor
from face_recognition.infrastructure.features.pose_landmark_normalizer import POSE_NORMALIZATION, PoseLandmarkNormalizer
from face_recognition.infrastructure.persistence.processed_dataset_repository import FilesystemProcessedDatasetRepository
from face_recognition.infrastructure.persistence.raw_dataset_reader import RawDatasetReader


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate all_xyz_pose_v2 in a new external output root; no overwrite")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        reader = RawDatasetReader(args.raw_root)
        repository = FilesystemProcessedDatasetRepository(args.output_root, reader.root,
            feature_representation=POSE_REPRESENTATION, normalized_representation=None,
            normalization_name="PoseLandmarkNormalizer", normalization_metadata=POSE_NORMALIZATION)
        result = DatasetProcessor(PoseLandmarkNormalizer, LandmarkFeatureExtractor()).process(reader, repository)
    except DatasetProcessingError as exc:
        print(f"Pose processing failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Pose processing interrupted; no completed output published.", file=sys.stderr)
        return 1
    except Exception:
        print("Pose processing failed: invalid source, geometry or output target. No completed output published.", file=sys.stderr)
        return 1
    print(f"POSE DATASET PROCESSING COMPLETE: {POSE_REPRESENTATION}")
    print(f"Participants={result.participants}; sessions={result.sessions}; conditions={result.conditions}")
    print(f"Samples={result.total_samples_processed}; rejected={result.rejected_samples}; errors={result.error_count}")
    print("478 landmarks / 1434 features; relative-depth face-local coordinates, not metric frontalization.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
