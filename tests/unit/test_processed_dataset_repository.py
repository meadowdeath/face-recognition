from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
import unittest

from face_recognition.application.process_dataset import DatasetProcessor, DatasetProcessingError
from face_recognition.domain.models.processed_sample import (
    BASELINE_REPRESENTATION, NORMALIZED_REPRESENTATION, REPRESENTATIONS, ProcessingIssue, ProcessingSummary,
)
from face_recognition.infrastructure.features.landmark_feature_extractor import LandmarkFeatureExtractor
from face_recognition.infrastructure.features.landmark_normalizer import LandmarkNormalizer
from face_recognition.infrastructure.persistence import processed_dataset_repository as persistence
from face_recognition.infrastructure.persistence.processed_dataset_repository import (
    FilesystemProcessedDatasetRepository, ProcessedDatasetPersistenceError,
)
from face_recognition.infrastructure.persistence.raw_dataset_reader import RawDatasetReader
from tests.processed_dataset_fakes import synthetic_sample, write_synthetic_session


class ProcessedDatasetRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory(prefix="synthetic_derived_repository_")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.raw = self.directory / "raw"
        self.raw.mkdir()
        write_synthetic_session(self.raw)
        self.output = self.directory / "processed"
        self.processor = DatasetProcessor(LandmarkNormalizer, LandmarkFeatureExtractor())

    def run_processing(self, output: Path | None = None):
        repository = FilesystemProcessedDatasetRepository(output or self.output, self.raw)
        summary = self.processor.process(RawDatasetReader(self.raw), repository)
        return repository, summary

    @staticmethod
    def snapshot(directory: Path) -> dict:
        return {path.relative_to(directory).as_posix():
                (path.read_bytes() if path.is_file() else None, path.stat().st_mtime_ns)
                for path in (directory, *directory.rglob("*"))}

    @staticmethod
    def output_bytes(directory: Path) -> dict:
        return {path.relative_to(directory).as_posix(): path.read_bytes()
                for path in directory.rglob("*") if path.is_file()}

    def test_representations_are_separate_and_preserve_values_order_and_provenance(self) -> None:
        self.run_processing()
        relative = Path("subject_test/visit_test/center.jsonl")
        normalized_path = self.output / NORMALIZED_REPRESENTATION / relative
        features_path = self.output / BASELINE_REPRESENTATION / relative
        normalized_records = [json.loads(line) for line in normalized_path.read_text(encoding="utf-8").splitlines()]
        feature_records = [json.loads(line) for line in features_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(normalized_records), 2)
        self.assertEqual(len(feature_records), 2)
        for index, (normalized_data, feature_data) in enumerate(zip(normalized_records, feature_records)):
            expected_normalized, expected_features = self.processor.process_sample(synthetic_sample(index=index))
            expected_points = [[p.x, p.y, p.z] for p in expected_normalized.landmarks.points]
            self.assertTrue(normalized_data["normalized_landmarks"] == expected_points,
                            "Normalized output must preserve all coordinates in original index order")
            self.assertTrue(tuple(feature_data["features"]) == expected_features.features.values,
                            "Baseline output must preserve extractor values and XYZ order")
            self.assertNotIn("features", normalized_data)
            self.assertNotIn("normalized_landmarks", feature_data)
            self.assertEqual(normalized_data["representation"], NORMALIZED_REPRESENTATION)
            self.assertEqual(feature_data["representation"], BASELINE_REPRESENTATION)
            self.assertEqual(feature_data["feature_count"], feature_data["landmark_count"] * 3)
            for record in (normalized_data, feature_data):
                self.assertEqual(record["schema_version"], 1)
                self.assertEqual(record["source_relative_path"], relative.as_posix())
                for name, value in asdict(expected_normalized.provenance).items():
                    self.assertEqual(record[name], value)

    def test_processing_manifests_have_correct_counts_and_completed_state(self) -> None:
        write_synthetic_session(self.raw, "another_test", "visit_a")
        write_synthetic_session(self.raw, "another_test", "visit_b")
        _, summary = self.run_processing()
        self.assertEqual(summary, ProcessingSummary(2, 3, 6, 12))
        root = json.loads((self.output / "processing_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(root["completion_state"], "completed")
        self.assertEqual(root["representations"], list(REPRESENTATIONS))
        self.assertEqual(root["total_condition_files"], 12)
        for representation in REPRESENTATIONS:
            directory = self.output / representation
            data = json.loads((directory / "processing_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(data["representation"], representation)
            self.assertEqual(data["completion_state"], "completed")
            self.assertEqual(data["condition_files"], 6)
            for name, value in asdict(summary).items():
                self.assertEqual(data[name], value)
            self.assertEqual(len(tuple(directory.rglob("*.jsonl"))), 6)

    def test_output_is_byte_deterministic_and_contains_no_absolute_paths(self) -> None:
        self.run_processing()
        second = self.directory / "processed_second"
        self.run_processing(second)
        first_bytes, second_bytes = self.output_bytes(self.output), self.output_bytes(second)
        self.assertTrue(first_bytes == second_bytes, "Repeated processing must serialize identically")
        for path in (self.output / "processing_manifest.json", *self.output.rglob("*.jsonl")):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn(str(self.directory), text)
            self.assertNotIn(self.directory.as_posix(), text)

    def test_source_bytes_entries_and_modification_times_are_unchanged_and_no_raw_writes(self) -> None:
        before = self.snapshot(self.raw)
        original_open = Path.open

        def guarded_open(path: Path, mode: str = "r", *args, **kwargs):
            if mode not in ("r", "rb") and path.resolve().is_relative_to(self.raw):
                raise AssertionError("Attempted write inside the raw tree")
            return original_open(path, mode, *args, **kwargs)

        with patch.object(Path, "open", guarded_open):
            self.run_processing()
        self.assertTrue(self.snapshot(self.raw) == before, "Raw source files and modification times must remain unchanged")

    def test_existing_completed_output_is_refused_and_never_modified(self) -> None:
        self.run_processing()
        before = self.output_bytes(self.output)
        with self.assertRaisesRegex(ProcessedDatasetPersistenceError, "already exists"):
            FilesystemProcessedDatasetRepository(self.output, self.raw)
        self.assertTrue(self.output_bytes(self.output) == before, "Existing output must never be overwritten")

    def test_even_existing_empty_output_directory_is_refused(self) -> None:
        self.output.mkdir()
        with self.assertRaisesRegex(ProcessedDatasetPersistenceError, "already exists"):
            FilesystemProcessedDatasetRepository(self.output, self.raw)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_equal_descendant_and_ancestor_output_roots_are_rejected_without_raw_changes(self) -> None:
        before = self.snapshot(self.raw)
        for output in (self.raw, self.raw / "derived", self.directory):
            with self.subTest(case=output.name), self.assertRaisesRegex(ProcessedDatasetPersistenceError, "overlap"):
                FilesystemProcessedDatasetRepository(output, self.raw)
        self.assertTrue(self.snapshot(self.raw) == before, "Overlap rejection must not change source files")

    def test_resolved_path_overlap_and_repository_local_output_are_rejected(self) -> None:
        alias = self.directory / "alias"
        original_resolve = Path.resolve

        def resolve(path: Path, *args, **kwargs):
            return self.raw / "derived" if path == alias else original_resolve(path, *args, **kwargs)

        with patch.object(Path, "resolve", resolve):
            with self.assertRaisesRegex(ProcessedDatasetPersistenceError, "overlap"):
                FilesystemProcessedDatasetRepository(alias, self.raw)
        simulated_repo = self.directory / "repo"
        with patch.object(persistence, "PROJECT_ROOT", simulated_repo):
            with self.assertRaisesRegex(ProcessedDatasetPersistenceError, "outside the Git repository"):
                FilesystemProcessedDatasetRepository(simulated_repo / "derived", self.raw)
        self.assertFalse(simulated_repo.exists())

    def test_wrong_configured_source_cannot_bypass_raw_tree_protection(self) -> None:
        other_raw = self.directory / "other_raw"
        other_raw.mkdir()
        before = self.snapshot(self.raw)
        # This path is disjoint from other_raw, but lies inside the ACTUAL reader root.
        repository = FilesystemProcessedDatasetRepository(self.raw / "derived", other_raw)
        with self.assertRaises(DatasetProcessingError) as caught:
            self.processor.process(RawDatasetReader(self.raw), repository)
        self.assertEqual(caught.exception.issue.category, "output_initialization_failed")
        self.assertIsNone(repository.staging_root)
        self.assertTrue(self.snapshot(self.raw) == before, "A mismatched source root must fail before any writes")

    def test_flush_and_close_failures_do_not_leave_the_other_stream_open(self) -> None:
        repository = FilesystemProcessedDatasetRepository(self.output, self.raw)
        first, second = Mock(), Mock()
        first.flush.side_effect = OSError("synthetic flush failure")
        first.close.side_effect = OSError("synthetic close failure")
        repository._streams = (first, second)
        with patch.object(persistence.os, "fsync"):
            with self.assertRaises(OSError):
                repository._close_streams()
        first.close.assert_called_once()
        second.close.assert_called_once()
        self.assertEqual(repository._streams, ())

    def test_unpublished_staging_is_distinct_until_complete(self) -> None:
        repository = FilesystemProcessedDatasetRepository(self.output, self.raw)
        repository.begin()
        normalized, baseline = self.processor.process_sample(synthetic_sample())
        repository.write_sample(normalized, baseline)
        self.assertFalse(self.output.exists())
        stage = repository.staging_root
        data = json.loads((stage / "processing_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(data["completion_state"], "in_progress")
        repository.complete(ProcessingSummary(1, 1, 1, 1))
        self.assertTrue(self.output.is_dir())
        self.assertFalse(stage.exists())

    def test_normalization_failure_leaves_failed_metadata_but_no_final_output(self) -> None:
        path = self.raw / "subject_test/visit_test/center.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        data = json.loads(lines[1])
        for index in (33, 133, 362, 263):
            data["landmarks"][index] = [0, 0, 0]
        lines[1] = json.dumps(data)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        before = self.snapshot(self.raw)
        repository = FilesystemProcessedDatasetRepository(self.output, self.raw)
        with self.assertRaises(DatasetProcessingError):
            self.processor.process(RawDatasetReader(self.raw), repository)
        self.assertFalse(self.output.exists())
        metadata = json.loads((repository.staging_root / "processing_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["completion_state"], "failed")
        self.assertEqual((metadata["total_samples_processed"], metadata["rejected_samples"], metadata["error_count"]), (1, 1, 1))
        self.assertEqual(metadata["failure"]["sample_index"], 1)
        self.assertEqual(metadata["failure"]["category"], "normalization_failed")
        self.assertTrue(self.snapshot(self.raw) == before, "Failure must not alter raw source files")

    def test_publication_failure_does_not_publish_or_claim_completed_output(self) -> None:
        repository = FilesystemProcessedDatasetRepository(self.output, self.raw)
        with patch.object(persistence.os, "rename", side_effect=OSError("synthetic publication failure")):
            with self.assertRaises(DatasetProcessingError) as caught:
                self.processor.process(RawDatasetReader(self.raw), repository)
        self.assertEqual(caught.exception.issue.category, "output_completion_failed")
        self.assertFalse(self.output.exists())
        data = json.loads((repository.staging_root / "processing_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(data["completion_state"], "failed")

    def test_duplicate_out_of_order_samples_and_wrong_completion_counts_are_rejected(self) -> None:
        repository = FilesystemProcessedDatasetRepository(self.output, self.raw)
        repository.begin()
        normalized, baseline = self.processor.process_sample(synthetic_sample())
        repository.write_sample(normalized, baseline)
        with self.assertRaisesRegex(ProcessedDatasetPersistenceError, "index order"):
            repository.write_sample(normalized, baseline)
        with self.assertRaisesRegex(ProcessedDatasetPersistenceError, "counts do not match"):
            repository.complete(ProcessingSummary())
        repository.fail(ProcessingSummary(1, 1, 1, 1, error_count=1), ProcessingIssue("test_failure"))
        self.assertFalse(self.output.exists())
