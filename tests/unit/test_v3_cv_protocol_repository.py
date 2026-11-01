from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from face_recognition.application.create_v3_cv_protocol import create_v3_cv_protocol, metadata_sha256
from face_recognition.domain.models.v3_cv_protocol import V3ProtocolError
from face_recognition.infrastructure.persistence.aggregate_report import AggregateReportError
from face_recognition.infrastructure.persistence.artifact_integrity import file_sha256
from face_recognition.infrastructure.persistence.v3_cv_protocol_repository import (
    read_v3_catalog, read_v3_cv_protocol, read_v3_experiment_specification, v3_protocol_to_json, write_v3_cv_protocol,
)
from tests.v3_protocol_fakes import v3_catalog_fixture


class V3ProtocolRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="synthetic_v3_repository_")
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.processed, self.spec_path, self.spec = v3_catalog_fixture(self.folder, known_count=2)
        self.root = self.processed / self.spec.representation
        self.header = self.root / "processing_manifest.json"
        self.output = self.folder / "splits" / "protocol.json"

    def plan(self):
        return create_v3_cv_protocol(read_v3_experiment_specification(self.spec_path), read_v3_catalog(self.processed))

    def publish(self, protocol=None):
        protocol = protocol or self.plan()
        write_v3_cv_protocol(protocol, self.output, (self.processed, self.spec_path))
        return protocol

    def test_roundtrip_deterministic_serialization_and_exact_metadata_hashes(self):
        protocol = self.publish()
        self.assertEqual(read_v3_cv_protocol(self.output), protocol)
        self.assertEqual(protocol.catalog.processing_manifest_sha256, file_sha256(self.header))
        self.assertEqual(protocol.specification_sha256, metadata_sha256(self.spec))
        self.assertEqual(protocol.catalog_sha256, metadata_sha256(protocol.catalog))
        self.assertEqual(self.output.read_text(encoding="utf-8"), v3_protocol_to_json(self.plan()))

    def test_catalog_and_publication_never_open_jsonl_payloads(self):
        original = Path.open
        opened = []

        def metadata_only(path, *args, **kwargs):
            if path.suffix == ".jsonl":
                raise AssertionError("Biometric payload access forbidden")
            opened.append(path.name)
            return original(path, *args, **kwargs)

        with patch.object(Path, "open", metadata_only):
            protocol = self.publish()
            self.assertEqual(read_v3_cv_protocol(self.output), protocol)
        self.assertIn("processing_manifest.json", opened)
        text = self.output.read_text(encoding="utf-8")
        self.assertNotIn("SYNTHETIC_PAYLOAD_DO_NOT_READ", text)
        self.assertNotIn(str(self.folder), text)
        def check(value):
            if isinstance(value, dict):
                self.assertFalse(set(value) & {"landmarks", "features", "scores", "predictions", "sample_index"})
                for child in value.values():
                    check(child)
            elif isinstance(value, list):
                for child in value:
                    check(child)
            elif isinstance(value, str):
                self.assertFalse(Path(value).is_absolute())
        check(json.loads(text))

    def test_filesystem_iteration_order_does_not_change_catalog_or_folds(self):
        expected = self.plan()
        original = Path.iterdir
        with patch.object(Path, "iterdir", lambda path: iter(reversed(tuple(original(path))))):
            self.assertEqual(self.plan(), expected)

    def test_incomplete_malformed_wrong_representation_and_error_counts_rejected(self):
        original = self.header.read_text(encoding="utf-8")
        metadata = json.loads(original)
        for change in ({"completion_state": "in_progress"}, {"representation": "normalized_landmarks_v1"},
                       {"error_count": 1}, {"rejected_samples": 1}, {"sessions": True}, {"conditions": 0}):
            self.header.write_text(json.dumps({**metadata, **change}), encoding="utf-8")
            with self.subTest(change=change), self.assertRaises(V3ProtocolError):
                self.plan()
        self.header.write_text(original[:-1] + ',"schema_version":1}', encoding="utf-8")
        with self.assertRaises(V3ProtocolError):
            self.plan()

    def test_missing_condition_file_and_empty_file_rejected(self):
        path = next(self.root.rglob("*.jsonl"))
        path.write_bytes(b"")
        with self.assertRaises(V3ProtocolError):
            self.plan()
        path.unlink()
        with self.assertRaises(V3ProtocolError):
            self.plan()

    def test_empty_extra_session_or_participant_is_never_ignored(self):
        for folder in (self.root / self.spec.known_ids[0] / "extra_capture", self.root / "unassigned"):
            folder.mkdir()
            try:
                with self.assertRaises(V3ProtocolError):
                    self.plan()
            finally:
                folder.rmdir()

    def test_strict_specification_rejects_missing_extra_or_ambiguous_fields(self):
        raw = asdict(self.spec)
        for bad in ({k: v for k, v in raw.items() if k != "unknown_ids"}, {**raw, "extra": 0},
                    {**raw, "known_ids": "known_id"}, {**raw, "schema_version": True},
                    {**raw, "representation": "different"}):
            self.spec_path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(V3ProtocolError):
                read_v3_experiment_specification(self.spec_path)

    def test_catalog_and_protocol_digests_detect_changes(self):
        protocol = self.publish()
        path = next(self.root.rglob("*.jsonl"))
        path.write_bytes(b"SYNTHETIC_DIFFERENT_LENGTH\n")
        changed = self.plan()
        self.assertNotEqual(protocol.catalog_sha256, changed.catalog_sha256)
        self.assertNotEqual(protocol.protocol_sha256, changed.protocol_sha256)
        # A metadata-only change also binds a different exact processing report.
        self.header.write_text(self.header.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        self.assertNotEqual(changed.catalog_sha256, self.plan().catalog_sha256)
        raw = json.loads(self.output.read_text(encoding="utf-8"))
        raw["catalog"]["condition_files"][0]["size_bytes"] += 1
        self.output.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaises(V3ProtocolError):
            read_v3_cv_protocol(self.output)

    def test_even_rehashed_illegal_roles_or_generation_policy_are_rejected(self):
        self.publish()
        original = self.output.read_text(encoding="utf-8")
        for field in ("inner", "generation"):
            raw = json.loads(original)
            if field == "inner":
                inner = raw["outer_folds"][0]["inner_splits"][0]
                inner["known_fit_sessions"] = [inner["known_validation_session"]]
            else:
                raw["generation"]["version"] = 2
            del raw["protocol_sha256"]
            raw["protocol_sha256"] = metadata_sha256(raw)
            self.output.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(V3ProtocolError):
                read_v3_cv_protocol(self.output)

    def test_missing_protocol_field_and_duplicate_json_key_rejected(self):
        self.publish()
        raw = json.loads(self.output.read_text(encoding="utf-8"))
        del raw["catalog_sha256"]
        self.output.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaises(V3ProtocolError):
            read_v3_cv_protocol(self.output)
        text = v3_protocol_to_json(self.plan())
        self.output.write_text(text.rstrip()[:-1] + ',"schema_version":1}', encoding="utf-8")
        with self.assertRaises(V3ProtocolError):
            read_v3_cv_protocol(self.output)

    def test_completed_output_overwrite_refused_without_modification(self):
        protocol = self.publish()
        before = self.output.read_bytes(), self.output.stat().st_mtime_ns
        with self.assertRaises(AggregateReportError):
            self.publish(protocol)
        self.assertEqual(before, (self.output.read_bytes(), self.output.stat().st_mtime_ns))
        self.assertEqual(tuple(self.output.parent.iterdir()), (self.output,))
