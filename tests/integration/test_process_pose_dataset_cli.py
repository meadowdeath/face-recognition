from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from face_recognition.domain.models.dataset import CONDITIONS
from face_recognition.domain.models.processed_sample import REPRESENTATIONS, POSE_REPRESENTATION
from face_recognition.infrastructure.features.pose_landmark_normalizer import POSE_NORMALIZATION
from face_recognition.infrastructure.persistence.feature_metadata_reader import scan_feature_record
from face_recognition.presentation.cli import process_pose_dataset as cli
from tests.processed_dataset_fakes import write_synthetic_session


class ProcessPoseDatasetCLITests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="synthetic_pose_processing_")
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.raw, self.output = self.folder / "raw", self.folder / "pose"
        self.raw.mkdir()
        self.session = write_synthetic_session(self.raw, count=478, completed=CONDITIONS)

    def invoke(self, output=None):
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.main(["--raw-root", str(self.raw), "--output-root", str(output or self.output)])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_fresh_pose_only_output_preserves_raw_metadata_order_and_baseline_defaults(self):
        before = {str(p.relative_to(self.raw)): (p.read_bytes(), p.stat().st_mtime_ns) for p in self.raw.rglob("*") if p.is_file()}
        code, stdout, stderr = self.invoke()
        self.assertEqual((code, stderr), (0, ""))
        self.assertIn("1434 features", stdout)
        self.assertNotIn("subject_test", stdout)
        manifest = json.loads((self.output / "processing_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["representations"], [POSE_REPRESENTATION])
        self.assertEqual(manifest["normalization_parameters"], POSE_NORMALIZATION)
        self.assertEqual(manifest["total_samples_processed"], 26)
        self.assertEqual(manifest["completion_state"], "completed")
        self.assertEqual(REPRESENTATIONS, ("normalized_landmarks_v1", "all_xyz_v1"))
        self.assertFalse((self.output / "all_xyz_v1").exists())
        self.assertFalse((self.output / "normalized_landmarks_v1").exists())
        for condition in CONDITIONS:
            rows = (self.output / POSE_REPRESENTATION / "subject_test" / "visit_test" / f"{condition}.jsonl").read_text(encoding="utf-8").splitlines()
            parsed = [scan_feature_record(row, representation=POSE_REPRESENTATION) for row in rows]
            self.assertEqual([p.provenance.sample_index for p in parsed], [0, 1])
            self.assertTrue(all(p.feature_count == 1434 and p.provenance.condition == condition for p in parsed))
            self.assertEqual([p.provenance.detector_timestamp_ms for p in parsed], [1000, 1250])
        after = {str(p.relative_to(self.raw)): (p.read_bytes(), p.stat().st_mtime_ns) for p in self.raw.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_deterministic_bytes_and_output_overwrite_refusal(self):
        self.assertEqual(self.invoke()[0], 0)
        other = self.folder / "pose_again"
        self.assertEqual(self.invoke(other)[0], 0)
        first = {p.relative_to(self.output): p.read_bytes() for p in self.output.rglob("*") if p.is_file()}
        second = {p.relative_to(other): p.read_bytes() for p in other.rglob("*") if p.is_file()}
        self.assertEqual(first, second)
        self.assertEqual(self.invoke()[0], 1)
        self.assertEqual(first, {p.relative_to(self.output): p.read_bytes() for p in self.output.rglob("*") if p.is_file()})

    def test_invalid_geometry_leaves_no_completed_output(self):
        path = self.session / "center.jsonl"
        rows = path.read_text(encoding="utf-8").splitlines()
        row = json.loads(rows[0])
        # Raw capture records use a landmarks array. Collapse the two eye means.
        for index in (33, 133, 362, 263): row["landmarks"][index] = [.5, .5, .01]
        path.write_text(json.dumps(row) + "\n" + "\n".join(rows[1:]) + "\n", encoding="utf-8")
        self.assertEqual(self.invoke()[0], 1)
        self.assertFalse(self.output.exists())

    def test_source_overlap_and_preexisting_baseline_directory_are_refused(self):
        self.assertEqual(self.invoke(self.raw / "candidate")[0], 1)
        baseline = self.folder / "historical_processed"
        baseline.mkdir(); (baseline / "evidence.json").write_text("immutable", encoding="utf-8")
        self.assertEqual(self.invoke(baseline)[0], 1)
        self.assertEqual((baseline / "evidence.json").read_text(encoding="utf-8"), "immutable")
