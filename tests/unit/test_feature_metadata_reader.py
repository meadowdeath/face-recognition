from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

from face_recognition.infrastructure.persistence import feature_metadata_reader as reader_module
from face_recognition.infrastructure.persistence.feature_metadata_reader import (
    FeatureMetadataReader, FeatureMetadataReadError, feature_metadata_from_json,
)
from tests.identity_split_fakes import write_feature_dataset
from tests.normalization_quality_fakes import snapshot


class FeatureMetadataReaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory(prefix="synthetic_feature_metadata_")
        self.addCleanup(self.temporary.cleanup)
        self.processed = Path(self.temporary.name) / "processed"
        self.root = write_feature_dataset(self.processed, {"subject_b": ("visit_b",), "subject_a": ("visit_a",)})
        self.path = self.root / "subject_a/visit_a/center.jsonl"

    def first_line(self) -> str:
        return self.path.read_text(encoding="utf-8").splitlines()[0]

    def test_only_metadata_is_returned_in_deterministic_order_with_no_source_writes(self) -> None:
        before = snapshot(self.processed)
        records = tuple(FeatureMetadataReader(self.processed).iter_metadata())
        self.assertEqual(len(records), 8)
        self.assertEqual(records[0].provenance.person_id, "subject_a")
        self.assertEqual([r.provenance.sample_index for r in records[:2]], [0, 1])
        self.assertTrue(all(set(asdict(record)) == {"provenance", "landmark_count", "feature_count"} for record in records))
        self.assertTrue(snapshot(self.processed) == before, "Metadata reads must preserve source bytes and modification times")

    def test_decoder_never_decodes_a_feature_array_or_full_record_as_a_value(self) -> None:
        line = self.first_line()
        original = reader_module._DECODER.raw_decode
        starts = []

        def scalar_only(document, index=0):
            starts.append(document[index])
            if document[index] in "[{":
                raise AssertionError("Whole array/object decoding is forbidden")
            return original(document, index)

        with patch.object(reader_module._DECODER, "raw_decode", side_effect=scalar_only):
            metadata = feature_metadata_from_json(line)
        self.assertEqual(metadata.feature_count, 9)
        self.assertFalse(set(starts) & {"[", "{"})

    def test_arbitrary_json_member_order_and_whitespace_are_supported(self) -> None:
        data = json.loads(self.first_line())
        reversed_data = dict(reversed(tuple(data.items())))
        result = feature_metadata_from_json(" \t" + json.dumps(reversed_data) + "\r\n")
        self.assertEqual(result, feature_metadata_from_json(self.first_line()))

    def test_malformed_arrays_dimensions_and_nonfinite_values_fail_without_echoing_payload(self) -> None:
        baseline = json.loads(self.first_line())
        for changes in ({"features": []}, {"features": [[0]]}, {"features": [True] * 9},
                        {"features": ["secret"] * 9}, {"features": [float("nan")] * 9},
                        {"features": [float("inf")] * 9}, {"feature_count": 8}, {"landmark_count": 2},
                        {"inference_width": 0}, {"schema_version": 999}, {"representation": "normalized_landmarks_v1"}):
            with self.subTest(field=next(iter(changes))), self.assertRaises(FeatureMetadataReadError):
                feature_metadata_from_json(json.dumps({**baseline, **changes}))
        marker = "SYNTHETIC_PRIVATE_VECTOR_PAYLOAD"
        with self.assertRaises(FeatureMetadataReadError) as caught:
            feature_metadata_from_json(marker)
        self.assertNotIn(marker, str(caught.exception))
        self.assertTrue(caught.exception.__suppress_context__)

    def test_duplicate_keys_and_trailing_data_are_rejected(self) -> None:
        line = self.first_line()
        for bad in (line[:-1] + ',"sample_index":0}', line + " trailing", line.replace('"features":[', '"features":[,')):
            with self.assertRaises(FeatureMetadataReadError):
                feature_metadata_from_json(bad)

    def test_incomplete_manifest_and_missing_files_are_rejected(self) -> None:
        header = self.root / "processing_manifest.json"
        data = json.loads(header.read_text(encoding="utf-8"))
        header.write_text(json.dumps({**data, "completion_state": "in_progress"}), encoding="utf-8")
        with self.assertRaises(FeatureMetadataReadError):
            FeatureMetadataReader(self.processed)
        header.write_text(json.dumps(data), encoding="utf-8")
        self.path.unlink()
        with self.assertRaises(FeatureMetadataReadError):
            tuple(FeatureMetadataReader(self.processed).iter_metadata())

    def test_record_metadata_and_sample_indices_must_match_the_file(self) -> None:
        baseline = self.path.read_text(encoding="utf-8")
        for changes in ({"person_id": "another"}, {"session_id": "another"}, {"condition": "translation_left"},
                        {"sample_index": 1}, {"source_relative_path": "../outside.jsonl"}):
            lines = baseline.splitlines()
            lines[0] = json.dumps({**json.loads(lines[0]), **changes})
            self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.subTest(field=next(iter(changes))), self.assertRaises(FeatureMetadataReadError):
                tuple(FeatureMetadataReader(self.processed).iter_metadata())

    def test_total_counts_and_unexpected_files_are_rejected(self) -> None:
        header = self.root / "processing_manifest.json"
        data = json.loads(header.read_text(encoding="utf-8"))
        header.write_text(json.dumps({**data, "total_samples_processed": data["total_samples_processed"] + 1}), encoding="utf-8")
        with self.assertRaises(FeatureMetadataReadError):
            tuple(FeatureMetadataReader(self.processed).iter_metadata())
        header.write_text(json.dumps(data), encoding="utf-8")
        (self.path.parent / "unexpected.jsonl").write_text("synthetic", encoding="utf-8")
        with self.assertRaises(FeatureMetadataReadError):
            tuple(FeatureMetadataReader(self.processed).iter_metadata())

    def test_inconsistent_session_dimensions_are_rejected(self) -> None:
        lines = self.path.read_text(encoding="utf-8").splitlines()
        data = json.loads(lines[1])
        data["inference_width"] += 2
        lines[1] = json.dumps(data)
        self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with self.assertRaises(FeatureMetadataReadError):
            tuple(FeatureMetadataReader(self.processed).iter_metadata())
