from dataclasses import replace
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from face_recognition.domain.models.processed_sample import POSE_REPRESENTATION
from face_recognition.domain.models.rejection import RejectionError
from face_recognition.infrastructure.features.pose_landmark_normalizer import POSE_NORMALIZATION
from face_recognition.infrastructure.persistence.feature_metadata_reader import scan_feature_record
from face_recognition.infrastructure.persistence.v3_pose_representation import load_paired_pose_payloads
from tests.v3_novelty_fakes import novelty_data, write_numeric_payloads


class V3PosePayloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="synthetic_pose_payloads_")
        self.addCleanup(self.temporary.cleanup)
        folder = Path(self.temporary.name)
        self.data, self.baseline, _ = novelty_data(folder, samples=7)
        self.protocol = write_numeric_payloads(self.data, self.baseline, self.data.protocol.specification)
        self.candidate = folder / "candidate"
        shutil.copytree(self.baseline / "all_xyz_v1", self.candidate / POSE_REPRESENTATION)
        for path in (self.candidate / POSE_REPRESENTATION).rglob("*.jsonl"):
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            for row in rows: row["representation"] = POSE_REPRESENTATION
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        header = self.candidate / POSE_REPRESENTATION / "processing_manifest.json"
        metadata = json.loads(header.read_text(encoding="utf-8"))
        metadata.update(representation=POSE_REPRESENTATION, normalization="PoseLandmarkNormalizer",
            normalization_parameters=POSE_NORMALIZATION, feature_extraction="LandmarkFeatureExtractor", inference_dimensions="per_source_sample")
        header.write_text(json.dumps(metadata), encoding="utf-8")
        root = {k: v for k, v in metadata.items() if k not in ("representation", "condition_files")}
        root.update(representations=[POSE_REPRESENTATION], condition_files_per_representation=metadata["condition_files"], total_condition_files=metadata["condition_files"])
        (self.candidate / "processing_manifest.json").write_text(json.dumps(root), encoding="utf-8")
        first = self.protocol.catalog.condition_files[0]
        self.first = self.candidate / POSE_REPRESENTATION / first.person_id / first.session_id / f"{first.condition}.jsonl"

    def load(self):
        return load_paired_pose_payloads(self.baseline, self.candidate, self.protocol)

    def edit_first(self, mutate):
        rows = [json.loads(line) for line in self.first.read_text(encoding="utf-8").splitlines()]
        mutate(rows)
        self.first.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    def test_strict_pair_roundtrip_same_samples_roles_width_but_distinct_catalogs(self):
        baseline, candidate, catalog, manifests = self.load()
        self.assertEqual(baseline.samples, candidate.samples)
        np.testing.assert_array_equal(baseline.X, candidate.X)
        self.assertEqual(candidate.X.shape[1], 1434)
        self.assertFalse(candidate.X.flags.writeable)
        self.assertEqual(candidate.protocol, baseline.protocol)
        self.assertEqual(catalog.representation, POSE_REPRESENTATION)
        self.assertNotEqual(catalog, self.protocol.catalog)
        self.assertEqual(len(manifests), 2)

    def test_default_feature_parser_still_rejects_candidate_records(self):
        line = self.first.read_text(encoding="utf-8").splitlines()[0]
        with self.assertRaises(ValueError): scan_feature_record(line)
        self.assertEqual(scan_feature_record(line, representation=POSE_REPRESENTATION).feature_count, 1434)

    def test_wrong_candidate_width_nonfinite_and_representation_rejected(self):
        original = self.first.read_bytes()
        for mutate in (
            lambda r: r[0].update(feature_count=1431, landmark_count=477, features=r[0]["features"][:-3]),
            lambda r: r[0]["features"].__setitem__(0, float("nan")),
            lambda r: r[0].update(representation="all_xyz_v1"),
        ):
            self.first.write_bytes(original)
            self.edit_first(mutate)
            with self.assertRaises(RejectionError): self.load()

    def test_missing_duplicate_or_unpaired_metadata_rejected(self):
        original = self.first.read_bytes()
        for mutate in (
            lambda r: r.pop(),
            lambda r: r.__setitem__(1, r[0]),
            lambda r: [row.update(detector_timestamp_ms=row["detector_timestamp_ms"] + 1) for row in r],
        ):
            self.first.write_bytes(original)
            self.edit_first(mutate)
            with self.assertRaises(RejectionError): self.load()

    def test_unexpected_catalog_file_or_invalid_protocol_rejected(self):
        extra = self.first.with_name("unexpected.jsonl")
        extra.write_text("not allowed", encoding="utf-8")
        with self.assertRaises(ValueError): self.load()
        extra.unlink()
        with self.assertRaises(ValueError):
            load_paired_pose_payloads(self.baseline, self.candidate, replace(self.protocol, protocol_sha256="0" * 64))

    def test_wrong_normalization_manifest_rejected_before_payloads(self):
        path = self.candidate / POSE_REPRESENTATION / "processing_manifest.json"
        row = json.loads(path.read_text(encoding="utf-8")); row["normalization"] = "LandmarkNormalizer"
        path.write_text(json.dumps(row), encoding="utf-8")
        with patch("face_recognition.infrastructure.persistence.v3_pose_representation.load_v3_payloads") as loader:
            with self.assertRaises(RejectionError): self.load()
            loader.assert_not_called()
