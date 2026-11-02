import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from face_recognition.infrastructure.persistence.v3_deployable_dataset import V3DeployableInputs
from tests.v3_deployable_fakes import deployable_fixture


class V3DeployableDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory(prefix="synthetic_v3_deployable_payloads_")
        cls.data, cls.config, cls.provenance, cls.root, cls.config_path, cls.protocol_path, cls.pose_path = deployable_fixture(Path(cls.temporary.name), numeric=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_strict_pose_loader_catalog_bytes_and_metadata_binding(self):
        inputs = V3DeployableInputs(self.root, self.config_path)
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
                with patch("face_recognition.infrastructure.persistence.v3_deployable_dataset.load_v3_payloads") as loader:
                    with self.assertRaises(ValueError): V3DeployableInputs(self.root, self.config_path)
                    loader.assert_not_called()
            finally: path.write_bytes(original)

    def test_modified_payload_missing_condition_or_wrong_width_rejected(self):
        path = next((self.root / "all_xyz_pose_v2").rglob("*.jsonl"))
        original = path.read_bytes()
        try:
            inputs = V3DeployableInputs(self.root, self.config_path)
            rows = path.read_text(encoding="utf-8").splitlines()
            row = json.loads(rows[0]); row["features"] = row["features"][:-3]; row["feature_count"] = 1431; row["landmark_count"] = 477
            path.write_text(json.dumps(row) + "\n" + "\n".join(rows[1:]) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError): inputs.load()
            path.unlink()
            with self.assertRaises(ValueError): V3DeployableInputs(self.root, self.config_path)
        finally: path.write_bytes(original)

    def test_mutated_pose_report_or_processing_manifest_rejected(self):
        header = self.root / "processing_manifest.json"
        for path in (self.pose_path, header):
            original = path.read_bytes()
            try:
                inputs = V3DeployableInputs(self.root, self.config_path)
                path.write_bytes(original + b"\n")
                with self.assertRaises(ValueError): inputs.verify()
            finally: path.write_bytes(original)
