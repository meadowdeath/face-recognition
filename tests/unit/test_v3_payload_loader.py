from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest.mock import patch

import numpy as np

from face_recognition.application.create_v3_cv_protocol import create_v3_cv_protocol
from face_recognition.domain.models.rejection import RejectionError
from face_recognition.infrastructure.persistence.v3_cv_protocol_repository import read_v3_catalog
from face_recognition.infrastructure.persistence.v3_payload_loader import load_v3_payloads
from tests.v3_novelty_fakes import novelty_data, write_numeric_payloads


class V3PayloadLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory(prefix="synthetic_v3_numeric_")
        cls.data, cls.processed, _ = novelty_data(Path(cls.temporary.name), samples=7)
        cls.protocol = write_numeric_payloads(cls.data, cls.processed, cls.data.protocol.specification)
        cls.first = cls.processed / cls.protocol.catalog.condition_files[0].source_relative_path

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @contextmanager
    def edited(self, text):
        original = self.first.read_bytes()
        self.first.write_text(text, encoding="utf-8")
        try:
            yield create_v3_cv_protocol(self.protocol.specification, read_v3_catalog(self.processed))
        finally:
            self.first.write_bytes(original)

    def test_strict_roundtrip_readonly_float32_labels_and_source_unchanged(self):
        before = self.first.read_bytes()
        loaded = load_v3_payloads(self.processed, self.protocol)
        self.assertEqual(loaded.X.dtype, np.float32)
        self.assertEqual(loaded.X.shape, (len(self.data.y), 1434))
        self.assertFalse(loaded.X.flags.writeable or loaded.y.flags.writeable)
        np.testing.assert_array_equal(loaded.y, self.data.y)
        np.testing.assert_allclose(loaded.X, self.data.X.round(4), atol=1e-6)
        self.assertEqual(self.first.read_bytes(), before)
        self.assertNotIn("array", repr(loaded))

    def test_catalog_and_processing_manifest_mismatch_rejected_before_payload(self):
        header = self.processed / "all_xyz_v1" / "processing_manifest.json"
        original = header.read_bytes()
        original_open = Path.open
        def no_payload(path, *args, **kwargs):
            if path.suffix == ".jsonl":
                raise AssertionError("Catalog rejection must precede payloads")
            return original_open(path, *args, **kwargs)
        try:
            header.write_bytes(original + b"\n")
            with patch.object(Path, "open", no_payload), self.assertRaises(RejectionError):
                load_v3_payloads(self.processed, self.protocol)
        finally:
            header.write_bytes(original)

    def test_wrong_width_nan_inf_and_float32_overflow_rejected(self):
        rows = self.first.read_text(encoding="utf-8").splitlines()
        first = json.loads(rows[0])
        for value in (float("nan"), float("inf"), -float("inf"), 1e100):
            changed = {**first, "features": [value, *first["features"][1:]]}
            with self.edited("\n".join([json.dumps(changed), *rows[1:]]) + "\n") as protocol:
                with self.assertRaises(RejectionError):
                    load_v3_payloads(self.processed, protocol)
        changed = {**first, "features": first["features"][:-3], "feature_count": 1431, "landmark_count": 477}
        with self.edited("\n".join([json.dumps(changed), *rows[1:]]) + "\n") as protocol:
            with self.assertRaises(RejectionError):
                load_v3_payloads(self.processed, protocol)

    def test_duplicate_missing_or_misordered_samples_rejected(self):
        rows = self.first.read_text(encoding="utf-8").splitlines()
        for changed in ([rows[0], rows[0], *rows[2:]], [rows[0], *rows[2:]], list(reversed(rows))):
            with self.edited("\n".join(changed) + "\n") as protocol:
                with self.assertRaises(RejectionError):
                    load_v3_payloads(self.processed, protocol)

    def test_wrong_identity_session_condition_representation_and_timestamps_rejected(self):
        rows = self.first.read_text(encoding="utf-8").splitlines()
        for change in ({"person_id": "unassigned"}, {"session_id": "another"}, {"condition": "near"},
                       {"representation": "normalized_landmarks_v1"}, {"detector_timestamp_ms": 0}, {"inference_width": 320}):
            second = {**json.loads(rows[1]), **change}
            changed = [rows[0], json.dumps(second), *rows[2:]]
            with self.edited("\n".join(changed) + "\n") as protocol:
                with self.assertRaises(RejectionError):
                    load_v3_payloads(self.processed, protocol)

    def test_aggregate_expected_count_and_unexpected_file_rejected(self):
        header = self.processed / "all_xyz_v1" / "processing_manifest.json"
        original = header.read_bytes()
        try:
            metadata = json.loads(original)
            metadata["total_samples_processed"] += 1
            header.write_text(json.dumps(metadata), encoding="utf-8")
            protocol = create_v3_cv_protocol(self.protocol.specification, read_v3_catalog(self.processed))
            with self.assertRaises(RejectionError):
                load_v3_payloads(self.processed, protocol)
        finally:
            header.write_bytes(original)
        extra = self.first.parent / "unexpected.jsonl"
        try:
            extra.write_text("SYNTHETIC", encoding="utf-8")
            with self.assertRaises(RejectionError):
                load_v3_payloads(self.processed, self.protocol)
        finally:
            extra.unlink()

    def test_labels_and_protocol_digest_mismatch_fail_closed(self):
        y = self.data.y.copy()
        y[0] = 1 - y[0]
        with self.assertRaises(RejectionError):
            replace(self.data, y=y)
        with self.assertRaises(RejectionError):
            load_v3_payloads(self.processed, replace(self.protocol, protocol_sha256="0" * 64))
