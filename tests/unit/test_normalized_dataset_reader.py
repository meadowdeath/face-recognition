from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

from face_recognition.application.evaluate_normalization import NormalizationQualityEvaluator
from face_recognition.domain.models.dataset import CONDITIONS
from face_recognition.domain.models.processed_sample import NORMALIZED_REPRESENTATION
from face_recognition.infrastructure.features.normalization_reference import center_interocular_distance
from face_recognition.infrastructure.persistence.normalization_quality_report import write_quality_report
from face_recognition.infrastructure.persistence.normalized_dataset_reader import NormalizedDatasetReader, NormalizedDatasetReadError
from face_recognition.infrastructure.persistence.raw_dataset_reader import RawDatasetReader
from tests.normalization_quality_fakes import process_quality_fixture, snapshot, write_quality_session


class NormalizedDatasetReaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory(prefix="synthetic_normalized_reader_")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.raw = self.directory / "raw"
        self.raw.mkdir()
        self.processed = self.directory / "processed"
        write_quality_session(self.raw, conditions=CONDITIONS[:2])
        process_quality_fixture(self.raw, self.processed)
        self.root = self.processed / NORMALIZED_REPRESENTATION
        self.header = self.root / "processing_manifest.json"
        self.condition = self.root / "synthetic_subject/synthetic_visit/center.jsonl"

    def test_catalog_and_sample_decoding_retain_provenance_without_writes(self) -> None:
        before = snapshot(self.processed)
        reader = NormalizedDatasetReader(self.processed)
        self.assertEqual(reader.discover_conditions(), (("synthetic_subject", "synthetic_visit", "center"),
                                                       ("synthetic_subject", "synthetic_visit", "translation_left")))
        samples = reader.read_condition(reader.discover_conditions()[0])
        self.assertEqual([sample.provenance.sample_index for sample in samples], [0, 1])
        self.assertTrue(all(len(sample.landmarks.points) == 400 for sample in samples))
        self.assertTrue(snapshot(self.processed) == before, "Reading normalized data must not write or touch timestamps")

    def test_invalid_schema_shapes_nonfinite_and_relative_provenance_are_rejected(self) -> None:
        baseline = self.condition.read_text(encoding="utf-8")
        changes = ({"schema_version": 999}, {"representation": "all_xyz_v1"}, {"normalized_landmarks": []},
                   {"normalized_landmarks": [[0, 1]]}, {"normalized_landmarks": [[float("nan"), 0, 0]], "landmark_count": 1},
                   {"normalized_landmarks": [[True, 0, 0]], "landmark_count": 1},
                   {"source_relative_path": "../outside.jsonl"}, {"inference_height": 0})
        reader = NormalizedDatasetReader(self.processed)
        for fields in changes:
            lines = baseline.splitlines()
            data = json.loads(lines[0])
            data.update(fields)
            lines[0] = json.dumps(data)
            self.condition.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.subTest(field=next(iter(fields))), self.assertRaises(NormalizedDatasetReadError):
                reader.read_condition(("synthetic_subject", "synthetic_visit", "center"))

    def test_malformed_json_never_appears_in_error_messages(self) -> None:
        marker = "SYNTHETIC_PRIVATE_COORDINATE_PAYLOAD"
        self.condition.write_text(marker, encoding="utf-8")
        with self.assertRaises(NormalizedDatasetReadError) as caught:
            NormalizedDatasetReader(self.processed).read_condition(("synthetic_subject", "synthetic_visit", "center"))
        self.assertNotIn(marker, str(caught.exception))
        self.assertTrue(caught.exception.__suppress_context__)

    def test_missing_file_and_incomplete_or_wrong_manifest_fail(self) -> None:
        baseline = json.loads(self.header.read_text(encoding="utf-8"))
        for fields in ({"completion_state": "failed"}, {"schema_version": True}, {"representation": "all_xyz_v1"},
                       {"total_samples_processed": -1}, {"error_count": 1}):
            self.header.write_text(json.dumps({**baseline, **fields}), encoding="utf-8")
            with self.subTest(field=next(iter(fields))), self.assertRaises(NormalizedDatasetReadError):
                NormalizedDatasetReader(self.processed)
        self.header.write_text(json.dumps(baseline), encoding="utf-8")
        self.condition.unlink()
        with self.assertRaisesRegex(NormalizedDatasetReadError, "catalog"):
            NormalizedDatasetReader(self.processed).discover_conditions()

    def test_symlink_paths_and_unsafe_keys_are_rejected(self) -> None:
        with patch.object(Path, "is_symlink", lambda path: path == self.root):
            with self.assertRaisesRegex(NormalizedDatasetReadError, "without symlinks"):
                NormalizedDatasetReader(self.processed)
        reader = NormalizedDatasetReader(self.processed)
        with self.assertRaises(NormalizedDatasetReadError):
            reader.read_condition(("../outside", "visit", "center"))

    def test_report_contains_aggregates_only_and_is_deterministic(self) -> None:
        report = NormalizationQualityEvaluator(center_interocular_distance).evaluate(
            RawDatasetReader(self.raw), NormalizedDatasetReader(self.processed),
        )
        target, second = self.directory / "quality.json", self.directory / "quality_again.json"
        write_quality_report(report, target, (self.raw, self.processed))
        write_quality_report(report, second, (self.raw, self.processed))
        self.assertTrue(target.read_bytes() == second.read_bytes(), "Aggregate reports must be deterministic")
        data = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(data, json.loads(json.dumps(asdict(report))))

        def check(value):
            if isinstance(value, dict):
                self.assertFalse(set(value) & {"landmarks", "normalized_landmarks", "features", "person_id", "session_id",
                                              "sample_index", "detector_timestamp_ms"})
                for item in value.values():
                    check(item)
            elif isinstance(value, list):
                for item in value:
                    check(item)
            elif isinstance(value, str):
                self.assertNotIn("synthetic_subject", value)
                self.assertNotIn("synthetic_visit", value)
        check(data)

    def test_report_cannot_overwrite_input_files_or_an_existing_report(self) -> None:
        report = NormalizationQualityEvaluator(center_interocular_distance).evaluate(
            RawDatasetReader(self.raw), NormalizedDatasetReader(self.processed),
        )
        before_raw, before_processed = snapshot(self.raw), snapshot(self.processed)
        for target in (self.raw / "quality.json", self.processed / "quality.json", self.header):
            with self.assertRaises(ValueError):
                write_quality_report(report, target, (self.raw, self.processed))
        target = self.directory / "quality.json"
        write_quality_report(report, target, (self.raw, self.processed))
        before = target.read_bytes()
        with self.assertRaises(ValueError):
            write_quality_report(report, target, (self.raw, self.processed))
        self.assertTrue(target.read_bytes() == before, "Existing aggregate reports must not be overwritten")
        self.assertTrue(snapshot(self.raw) == before_raw, "Report creation must not alter raw input")
        self.assertTrue(snapshot(self.processed) == before_processed, "Report creation must not alter processed input")

    def test_failed_report_publication_leaves_no_partial_final_file(self) -> None:
        report = NormalizationQualityEvaluator(center_interocular_distance).evaluate(
            RawDatasetReader(self.raw), NormalizedDatasetReader(self.processed),
        )
        target = self.directory / "quality.json"
        with patch("face_recognition.infrastructure.persistence.normalization_quality_report.os.link", side_effect=OSError("test")):
            with self.assertRaises(OSError):
                write_quality_report(report, target, (self.raw, self.processed))
        self.assertFalse(target.exists())
        self.assertEqual(list(self.directory.glob("quality.json.*.tmp")), [])
