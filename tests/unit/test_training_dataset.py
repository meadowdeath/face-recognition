import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from face_recognition.domain.models.rejection import RejectionError
from face_recognition.infrastructure.persistence.training_dataset import TrainingInputs
from tests.recognition_fakes import deployable_fixture


class V3DeployableDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory(prefix="synthetic_v3_deployable_payloads_")
        cls.data, cls.config, cls.provenance, cls.root, cls.config_path, cls.protocol_path, cls.pose_path = deployable_fixture(Path(cls.temporary.name), numeric=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_strict_pose_loader_catalog_bytes_and_metadata_binding(self):
        inputs = TrainingInputs(self.root, self.config_path)
        data, provenance = inputs.load()
        self.assertEqual(provenance, self.provenance)
        self.assertEqual(data.samples, self.data.samples)
        self.assertEqual(data.representation, "all_xyz_pose_v2")
        self.assertEqual(data.X.shape[1], 1434)
        self.assertFalse(data.X.flags.writeable)
        inputs.verify()

    def test_bad_frozen_config_parameters_or_protocol_identity_metadata_rejected_before_payloads(self):
        for path, edit in ((self.config_path, lambda r: r["pipeline"].update(selected_features=512)),
                           (self.protocol_path, lambda r: r["specification"]["known_ids"].__setitem__(0, "unregistered"))):
            original = path.read_bytes()
            try:
                row = json.loads(original); edit(row)
                path.write_text(json.dumps(row), encoding="utf-8")
                with patch("face_recognition.infrastructure.persistence.training_dataset.load_training_payloads") as loader:
                    with self.assertRaises(ValueError): TrainingInputs(self.root, self.config_path)
                    loader.assert_not_called()
            finally: path.write_bytes(original)

    def test_modified_payload_missing_condition_or_wrong_width_rejected(self):
        path = next((self.root / "all_xyz_pose_v2").rglob("*.jsonl"))
        original = path.read_bytes()
        try:
            inputs = TrainingInputs(self.root, self.config_path)
            rows = path.read_text(encoding="utf-8").splitlines()
            row = json.loads(rows[0]); row["features"] = row["features"][:-3]; row["feature_count"] = 1431; row["landmark_count"] = 477
            path.write_text(json.dumps(row) + "\n" + "\n".join(rows[1:]) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError): inputs.load()
            path.unlink()
            with self.assertRaises(ValueError): TrainingInputs(self.root, self.config_path)
        finally: path.write_bytes(original)

    def test_mutated_pose_report_or_processing_manifest_rejected(self):
        header = self.root / "processing_manifest.json"
        for path in (self.pose_path, header):
            original = path.read_bytes()
            try:
                inputs = TrainingInputs(self.root, self.config_path)
                path.write_bytes(original + b"\n")
                with self.assertRaises(ValueError): inputs.verify()
            finally: path.write_bytes(original)

    def test_payload_nonfinite_wrong_roles_indices_counts_and_dimensions_fail_closed(self):
        path = next((self.root / "all_xyz_pose_v2").rglob("*.jsonl"))
        original = path.read_bytes()
        lines = original.decode("utf-8").splitlines()
        edits = (
            lambda r: r["features"].__setitem__(0, float("nan")),
            lambda r: r["features"].__setitem__(0, float("inf")),
            lambda r: r["features"].__setitem__(0, True),
            lambda r: r.update(person_id="unexpected"),
            lambda r: r.update(session_id="unexpected"),
            lambda r: r.update(condition="far" if r["condition"] != "far" else "near"),
            lambda r: r.update(sample_index=1),
            lambda r: r.update(representation="all_xyz_v1"),
            lambda r: r.update(inference_width=0),
        )
        try:
            for edit in edits:
                inputs = TrainingInputs(self.root, self.config_path)
                row = json.loads(lines[0]); edit(row)
                path.write_text(json.dumps(row) + "\n" + "\n".join(lines[1:]) + "\n", encoding="utf-8")
                with self.subTest(edit=edits.index(edit)), self.assertRaises(RejectionError): inputs.load()
                path.write_bytes(original)
            for changed in (lines[1:], [lines[0], *lines], list(reversed(lines))):
                inputs = TrainingInputs(self.root, self.config_path)
                path.write_text("\n".join(changed) + "\n", encoding="utf-8")
                with self.assertRaises(RejectionError): inputs.load()
                path.write_bytes(original)
        finally:
            path.write_bytes(original)

    def test_unexpected_catalog_file_and_incomplete_manifest_are_rejected(self):
        extra = self.root / "all_xyz_pose_v2" / "unexpected"
        extra.mkdir()
        try:
            with self.assertRaises(RejectionError): TrainingInputs(self.root, self.config_path)
        finally: extra.rmdir()
        manifest = self.root / "all_xyz_pose_v2" / "processing_manifest.json"
        original = manifest.read_bytes()
        try:
            row = json.loads(original); row["completion_state"] = "in_progress"
            manifest.write_text(json.dumps(row), encoding="utf-8")
            with self.assertRaises(RejectionError): TrainingInputs(self.root, self.config_path)
        finally: manifest.write_bytes(original)
