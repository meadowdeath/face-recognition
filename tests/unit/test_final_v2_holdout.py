from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from face_recognition.application.final_evaluation_data import FinalEvaluationError
from face_recognition.infrastructure.persistence.final_v2_holdout_repository import (
    FinalV2HoldoutLoader, read_final_v2_manifest, read_v2_final_inputs, write_final_v2_manifest,
)
from face_recognition.infrastructure.persistence.v2_model_repository import read_v2_model_metadata, v2_model_fingerprints
from face_recognition.presentation.cli.create_final_v2_holdout import main as create_main
from face_recognition.presentation.cli.evaluate_final_v2_open_set import main as evaluate_main
from tests.final_v2_fakes import final_v2_fixture


class FinalV2HoldoutValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = TemporaryDirectory(prefix="synthetic_final_v2_original_")
        cls.original_root = Path(cls.original.name)
        cls.original_paths = final_v2_fixture(cls.original_root)

    @classmethod
    def tearDownClass(cls):
        cls.original.cleanup()

    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="synthetic_final_v2_case_")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / "study"
        shutil.copytree(self.original_root, self.directory)  # Synthetic files only, wholly within this test's temp directory.
        self.paths = tuple(self.directory / p.relative_to(self.original_root) for p in self.original_paths)
        self.root, self.source, self.unknown, self.model, self.config, self.development = self.paths
        self.inputs = self.read_inputs()
        self.loader = FinalV2HoldoutLoader(self.root)
        self.manifest = self.loader.catalog_manifest(self.inputs)
        self.manifest_path = self.directory / "final_holdout.json"
        write_final_v2_manifest(self.manifest, self.manifest_path, self.paths)

    def read_inputs(self):
        return read_v2_final_inputs(self.source, self.unknown, self.config, self.development, self.model)

    def metadata_path(self):
        return self.root / "all_xyz_v1" / "processing_manifest.json"

    def feature_path(self):
        return next((self.root / "all_xyz_v1").rglob("*.jsonl"))

    def test_catalog_roundtrip_deterministic_no_feature_inspection(self):
        before = self.manifest_path.read_bytes()
        manifest, digest = read_final_v2_manifest(self.manifest_path)
        self.assertEqual(manifest, self.manifest)
        with patch.object(self.loader, "_read_references", side_effect=AssertionError("No payload")):
            self.loader.authorize(manifest, self.inputs)
        self.assertEqual(self.manifest_path.read_bytes(), before)
        self.assertEqual(len(digest), 64)
        self.assertNotIn(str(self.directory), before.decode())

    def test_unexpected_processed_person_or_session_is_never_ignored(self):
        extra = self.root / "all_xyz_v1" / "unexpected_person" / "unlisted_visit"
        extra.mkdir(parents=True)
        (extra / "center.jsonl").write_text("DO NOT OPEN")
        with patch.object(self.loader, "_read_references") as read:
            with self.assertRaises(ValueError):
                self.loader.load_final(self.manifest, self.inputs)
            read.assert_not_called()

    def test_missing_referenced_session_rejected_without_payload(self):
        session = self.manifest.known_sessions[0]
        old = self.root / "all_xyz_v1" / session.person_id / session.session_id
        old.rename(old.with_name("wrong_visit"))
        with patch.object(self.loader, "_read_references") as read:
            with self.assertRaises(ValueError):
                self.loader.load_final(self.manifest, self.inputs)
            read.assert_not_called()

    def test_missing_condition_file_rejected(self):
        self.feature_path().unlink()
        with self.assertRaises(ValueError):
            self.loader.authorize(self.manifest, self.inputs)

    def test_incomplete_processed_dataset_rejected(self):
        p = self.metadata_path()
        metadata = json.loads(p.read_text())
        metadata["completion_state"] = "incomplete"
        p.write_text(json.dumps(metadata))
        with self.assertRaises(ValueError):
            FinalV2HoldoutLoader(self.root)

    def test_duplicate_processing_metadata_keys_rejected_before_payload(self):
        path = self.metadata_path()
        path.write_text(path.read_text().strip()[:-1] + ', "completion_state": "completed"}')
        with patch("joblib.load", side_effect=AssertionError("No estimator deserialization")), \
                patch.object(self.loader, "_read_references") as read:
            with self.assertRaises(ValueError):
                FinalV2HoldoutLoader(self.root).catalog_manifest(self.inputs)
            read.assert_not_called()

    def test_wrong_representation_rejected(self):
        with self.assertRaises(FinalEvaluationError):
            self.loader.authorize(replace(self.manifest, representation="normalized_landmarks_v1"), self.inputs)
        p = self.metadata_path()
        metadata = json.loads(p.read_text())
        metadata["representation"] = "normalized_landmarks_v1"
        p.write_text(json.dumps(metadata))
        with self.assertRaises(ValueError):
            FinalV2HoldoutLoader(self.root)

    def test_wrong_feature_dimension_rejected_before_inference(self):
        path = self.feature_path()
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["features"] = rows[0]["features"][:-3]
        rows[0]["feature_count"] -= 3
        rows[0]["landmark_count"] -= 1
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        with self.assertRaises(FinalEvaluationError):
            self.loader.load_final(self.manifest, self.inputs)

    def test_malformed_metadata_and_sample_order_fail_explicitly(self):
        path = self.feature_path()
        original = path.read_text()
        for field, value in (("person_id", "wrong_person"), ("session_id", "wrong_visit"), ("sample_index", 99), ("inference_width", 0)):
            rows = [json.loads(line) for line in original.splitlines()]
            rows[0][field] = value
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            with self.assertRaises(ValueError):
                self.loader.load_final(self.manifest, self.inputs)

    def test_malformed_json_and_nonfinite_feature_fail(self):
        path = self.feature_path()
        original = path.read_text()
        path.write_text("NOT JSON")
        with self.assertRaises(ValueError):
            self.loader.load_final(self.manifest, self.inputs)
        for value in (float("nan"), float("inf"), float("-inf")):
            rows = [json.loads(line) for line in original.splitlines()]
            rows[0]["features"][0] = value
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            with self.assertRaises(ValueError):
                self.loader.load_final(self.manifest, self.inputs)

    def test_changed_development_report_or_config_provenance_rejected(self):
        original = self.development.read_bytes()
        self.development.write_bytes(original + b"\n")
        with self.assertRaises(ValueError):
            self.read_inputs()
        self.development.write_bytes(original)
        self.config.write_bytes(self.config.read_bytes() + b"\n")
        with self.assertRaises(ValueError):
            self.read_inputs()

    def test_tampered_model_detected_before_unpickle(self):
        artifact = self.model / "scaler.joblib"
        artifact.write_bytes(artifact.read_bytes() + b"tampered")
        with patch("joblib.load", side_effect=AssertionError("No unpickle")) as read:
            with self.assertRaises(ValueError):
                self.read_inputs()
            read.assert_not_called()

    def test_changed_processing_metadata_cannot_reuse_manifest(self):
        p = self.metadata_path()
        p.write_text(p.read_text() + "\n")
        with self.assertRaises(FinalEvaluationError):
            FinalV2HoldoutLoader(self.root).authorize(self.manifest, self.inputs)

    def test_manifest_digest_and_strict_schema_no_silent_defaults(self):
        original = json.loads(self.manifest_path.read_text())
        for key, value in (("schema_version", True), ("original_feature_count", 1434.), ("completion_state", "complete")):
            data = dict(original, **{key: value})
            self.manifest_path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                read_final_v2_manifest(self.manifest_path)

    def test_manifest_overwrite_refused(self):
        before = self.manifest_path.read_bytes()
        with self.assertRaises(ValueError):
            write_final_v2_manifest(self.manifest, self.manifest_path, self.paths)
        self.assertEqual(before, self.manifest_path.read_bytes())


if __name__ == "__main__":
    unittest.main()
