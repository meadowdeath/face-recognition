from contextlib import ExitStack
from dataclasses import asdict, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from face_recognition.application.create_identity_split import IdentitySplitPlanner
from face_recognition.application.validate_identity_split import SplitValidationError
from face_recognition.domain.models.identity_split import IdentityPartition
from face_recognition.infrastructure.persistence.development_dataset_loader import DevelopmentDatasetLoader, DevelopmentDatasetReadError
from face_recognition.infrastructure.persistence.feature_metadata_reader import FeatureMetadataReader
from face_recognition.infrastructure.persistence.identity_split_repository import IdentitySplitPersistenceError, read_identity_split
from tests.training_fold_fakes import training_fixture


class DevelopmentDatasetLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="synthetic_training_")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.processed, self.root, self.manifest, self.split_path = training_fixture(self.directory)
        self.path = self.root / "known_alpha" / "visit_a" / "center.jsonl"

    def load(self):
        return DevelopmentDatasetLoader(self.processed).load(read_identity_split(self.split_path))

    def mutate(self, change):
        records = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]
        change(records)
        self.path.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")

    def test_manifest_round_trip_and_deterministic_read_only_loading(self):
        self.assertEqual(read_identity_split(self.split_path), self.manifest)
        before = {path: (path.read_bytes(), path.stat().st_mtime_ns)
                  for path in self.directory.rglob("*") if path.is_file()}
        first, second = self.load(), self.load()
        self.assertEqual(first.samples, second.samples)
        self.assertTrue(np.array_equal(first.X, second.X), "Deterministic development matrix")
        self.assertTrue(np.array_equal(first.y, second.y), "Deterministic labels")
        self.assertEqual(first.label_to_person, ("known_alpha", "known_beta"))
        self.assertEqual({path: (path.read_bytes(), path.stat().st_mtime_ns)
                          for path in self.directory.rglob("*") if path.is_file()}, before)
        self.assertEqual(first.X.shape, (16, 9))
        self.assertEqual(first.X.dtype, np.float32)
        self.assertEqual(first.y.dtype, np.int32)
        self.assertEqual(first.X.nbytes, 16 * 9 * 4)
        self.assertFalse(first.X.flags.writeable)
        self.assertFalse(first.y.flags.writeable)
        self.assertNotIn("array", repr(first))

    def test_feature_label_and_provenance_association(self):
        self.mutate(lambda rows: rows[1]["features"].__setitem__(0, 17.25))
        data = self.load()
        for index, sample in enumerate(data.samples):
            self.assertEqual(data.label_to_person[int(data.y[index])], sample.person_id)
            self.assertIs(sample.partition, IdentityPartition.KNOWN_TRAIN_POOL)
            if (sample.person_id, sample.session_id, sample.condition, sample.sample_index) == ("known_alpha", "visit_a", "center", 1):
                self.assertTrue(data.X[index, 0] == 17.25, "Feature and provenance association")
        keys = [(s.session_id, s.person_id) for s in data.samples]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual({s.condition for s in data.samples}, {"center", "translation_left"})
        self.assertEqual({s.sample_index for s in data.samples}, {0, 1})

    def test_protected_payloads_are_never_opened_even_for_validation(self):
        protected = {self.processed / row.source_relative_path for row in self.manifest.samples
                     if row.partition is not IdentityPartition.KNOWN_TRAIN_POOL}
        for path in protected:
            path.write_text("deliberately unreadable protected payload", encoding="utf-8")
        original_open = Path.open
        opened = []

        def guarded_open(path, *args, **kwargs):
            self.assertNotIn(path, protected, "Protected payload must not be opened")
            opened.append(path)
            return original_open(path, *args, **kwargs)

        with ExitStack() as stack:
            stack.enter_context(patch.object(Path, "open", guarded_open))
            stack.enter_context(patch.object(FeatureMetadataReader, "iter_metadata", side_effect=AssertionError("Full scan forbidden")))
            stack.enter_context(patch.object(IdentitySplitPlanner, "plan", side_effect=AssertionError("Replanning forbidden")))
            data = self.load()
        self.assertEqual(len(data.samples), 16)
        payloads = {path for path in opened if path.suffix == ".jsonl"}
        self.assertEqual(payloads, {self.processed / row.source_relative_path for row in data.samples})

    def test_protected_partition_requests_are_rejected_before_payload_read(self):
        loader = DevelopmentDatasetLoader(self.processed)
        for partition in (IdentityPartition.KNOWN_TEST, IdentityPartition.UNKNOWN_POOL, "known_train_pool"):
            with self.subTest(partition=partition), patch.object(Path, "open", side_effect=AssertionError("No read")):
                with self.assertRaisesRegex(DevelopmentDatasetReadError, "Only known_train_pool"):
                    loader.load(self.manifest, partition)

    def test_missing_referenced_sample_is_rejected(self):
        self.mutate(lambda rows: rows.pop())
        with self.assertRaisesRegex(DevelopmentDatasetReadError, "known_alpha/visit_a/center"):
            self.load()

    def test_unexpected_extra_sample_is_rejected(self):
        self.mutate(lambda rows: rows.append({**rows[-1], "sample_index": 2}))
        with self.assertRaises(DevelopmentDatasetReadError):
            self.load()

    def test_missing_or_unexpected_condition_file_is_rejected(self):
        self.path.rename(self.path.with_name("near.jsonl"))
        with self.assertRaisesRegex(DevelopmentDatasetReadError, "catalog"):
            self.load()

    def test_metadata_mismatches_and_invalid_dimensions_are_rejected(self):
        original = self.path.read_bytes()
        for field, value in (("person_id", "different"), ("session_id", "held_out"), ("condition", "near"),
                             ("sample_index", 9), ("inference_width", 0), ("detector_timestamp_ms", -1)):
            with self.subTest(field=field):
                self.path.write_bytes(original)
                self.mutate(lambda rows: rows[0].__setitem__(field, value))
                with self.assertRaises(DevelopmentDatasetReadError):
                    self.load()

    def test_feature_length_inconsistency_is_rejected(self):
        def change(rows):
            rows[1]["features"] += [0.0, 0.0, 0.0]
            rows[1]["feature_count"] += 3
            rows[1]["landmark_count"] += 1
        self.mutate(change)
        with self.assertRaises(DevelopmentDatasetReadError):
            self.load()

    def test_nonfinite_and_float32_overflow_are_rejected_without_value_exposure(self):
        original = self.path.read_bytes()
        for value in (float("nan"), float("inf"), -float("inf"), 1.23456789e100):
            with self.subTest(case=str(type(value))):
                self.path.write_bytes(original)
                self.mutate(lambda rows: rows[1]["features"].__setitem__(0, value))
                with self.assertRaises(DevelopmentDatasetReadError) as caught:
                    self.load()
                self.assertNotIn(str(value), str(caught.exception))
                self.assertNotIn("features", str(caught.exception))

    def test_timestamp_and_within_session_dimension_inconsistency_are_rejected(self):
        original = self.path.read_bytes()
        for field, value in (("detector_timestamp_ms", 1000), ("inference_height", 271)):
            with self.subTest(field=field):
                self.path.write_bytes(original)
                self.mutate(lambda rows: rows[1].__setitem__(field, value))
                with self.assertRaises(DevelopmentDatasetReadError):
                    self.load()

    def test_malformed_json_is_sanitized(self):
        self.path.write_text('{"features": [987654.321', encoding="utf-8")
        with self.assertRaises(DevelopmentDatasetReadError) as caught:
            self.load()
        self.assertNotIn("987654", str(caught.exception))

    def test_arbitrary_identity_condition_sample_and_feature_counts(self):
        processed, _, manifest, _ = training_fixture(self.directory / "other", known=("single",),
                                                     conditions=("center", "near", "pitch_down"), samples=5, landmarks=7)
        data = DevelopmentDatasetLoader(processed).load(manifest)
        self.assertEqual(data.X.shape, (30, 21))
        self.assertEqual(data.label_to_person, ("single",))

    def test_changed_processing_completion_or_counts_are_rejected(self):
        path = self.root / "processing_manifest.json"
        original = path.read_text(encoding="utf-8")
        for field, value in (("completion_state", "failed"), ("total_samples_processed", 1)):
            with self.subTest(field=field):
                payload = json.loads(original)
                payload[field] = value
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(DevelopmentDatasetReadError):
                    self.load()

    def test_tampered_manifest_is_rejected_before_any_feature_read(self):
        original = asdict(self.manifest)
        mutations = (
            lambda d: d.__setitem__("completion_state", "failed"),
            lambda d: d.__setitem__("total_sample_count", 1),
            lambda d: d["samples"][0].__setitem__("partition", "unknown_pool"),
            lambda d: d["samples"][0].__setitem__("source_relative_path", "../escape.jsonl"),
            lambda d: d["samples"][0].__setitem__("sample_index", 1),
            lambda d: d["samples"].reverse(),
            lambda d: d["session_assignments"].pop(),
            lambda d: d["partition_counts"][0].__setitem__("samples", True),
            lambda d: d["specification"]["known_ids"].reverse(),
            lambda d: d.__setitem__("features", [123.456]),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(case=index):
                data = json.loads(json.dumps(original))
                mutate(data)
                self.split_path.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaises(IdentitySplitPersistenceError):
                    read_identity_split(self.split_path)

    def test_direct_manifest_cannot_relabel_protected_samples(self):
        protected = next(row for row in self.manifest.samples if row.partition is IdentityPartition.KNOWN_TEST)
        changed = replace(self.manifest, samples=tuple(
            replace(row, partition=IdentityPartition.KNOWN_TRAIN_POOL) if row == protected else row
            for row in self.manifest.samples))
        loader = DevelopmentDatasetLoader(self.processed)
        with patch.object(Path, "open", side_effect=AssertionError("No payload reads")):
            with self.assertRaises(SplitValidationError):
                loader.load(changed)

    def test_duplicate_json_keys_in_split_are_rejected(self):
        text = self.split_path.read_text(encoding="utf-8")
        self.split_path.write_text(text.replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1'), encoding="utf-8")
        with self.assertRaises(IdentitySplitPersistenceError):
            read_identity_split(self.split_path)
