from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from face_recognition.infrastructure.persistence.v3_cv_protocol_repository import write_v3_cv_protocol
from face_recognition.presentation.cli import evaluate_v3_pose_representation as cli
from tests.pose_representation_fakes import paired_pose_data
from tests.v3_mahalanobis_fakes import RecordingSameModelRunner


class V3PoseComparisonCLITests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="synthetic_pose_cli_")
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.baseline, self.candidate, self.catalog, self.baseline_root, specification = paired_pose_data(self.folder)
        self.candidate_root = self.folder / "pose"
        self.protocol = self.folder / "protocol.json"
        write_v3_cv_protocol(self.baseline.protocol, self.protocol, (self.baseline_root, specification))
        self.output = self.folder / "reports" / "comparison.json"

    def invoke(self, output=None):
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.main(["--baseline-root", str(self.baseline_root), "--candidate-root", str(self.candidate_root),
                "--protocol", str(self.protocol), "--report", str(output or self.output)])
        return code, stdout.getvalue(), stderr.getvalue()

    def loader_patches(self):
        # Numeric populations are synthetic; no actual feature file is opened.
        return patch.object(cli, "load_paired_pose_payloads", return_value=(self.baseline, self.candidate, self.catalog, ("a", "b")))

    def test_success_includes_pitch_down_and_all_conditions_deterministically(self):
        before = self.protocol.read_bytes(), self.protocol.stat().st_mtime_ns
        def catalog(root, **kwargs):
            return self.catalog if kwargs else self.baseline.protocol.catalog
        with self.loader_patches(), patch.object(cli, "SklearnV3SameModelRunner", side_effect=RecordingSameModelRunner), \
                patch.object(cli, "read_v3_catalog", side_effect=catalog), patch.object(cli, "pose_processing_fingerprints", return_value=("a", "b")):
            code, stdout, stderr = self.invoke()
            self.assertEqual((code, stderr), (0, ""))
            self.assertEqual(self.invoke(self.folder / "second.json")[0], 0)
        self.assertEqual(self.output.read_bytes(), (self.folder / "second.json").read_bytes())
        self.assertIn("pitch_down:", stdout)
        self.assertIn("yaw_left:", stdout)
        self.assertIn("all_xyz_v1", stdout)
        self.assertIn("all_xyz_pose_v2", stdout)
        self.assertIn("iterative study", stdout)
        self.assertNotIn(str(self.folder), stdout)
        self.assertNotIn("known_00", stdout)
        self.assertEqual(before, (self.protocol.read_bytes(), self.protocol.stat().st_mtime_ns))

    def test_overwrite_and_protocol_alias_rejected_before_payloads(self):
        self.output.parent.mkdir()
        self.output.write_text("immutable", encoding="utf-8")
        for target in (self.output, self.protocol):
            before = target.read_bytes()
            with patch.object(cli, "load_paired_pose_payloads") as loader:
                self.assertEqual(self.invoke(target)[0], 1)
                loader.assert_not_called()
            self.assertEqual(target.read_bytes(), before)

    def test_invalid_protocol_is_rejected_before_loading(self):
        self.protocol.write_text("{}", encoding="utf-8")
        with patch.object(cli, "load_paired_pose_payloads") as loader:
            self.assertEqual(self.invoke()[0], 1)
            loader.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_loading_or_fitting_failure_publishes_no_final_report(self):
        with patch.object(cli, "load_paired_pose_payloads", side_effect=ValueError("SYNTHETIC_PRIVATE_DATA")):
            code, stdout, stderr = self.invoke()
        self.assertEqual(code, 1)
        self.assertNotIn("SYNTHETIC_PRIVATE_DATA", stdout + stderr)
        self.assertFalse(self.output.exists())
        runner = RecordingSameModelRunner()
        runner.fit = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("SYNTHETIC_PRIVATE_DATA"))
        with self.loader_patches(), patch.object(cli, "SklearnV3SameModelRunner", return_value=runner):
            code, stdout, stderr = self.invoke()
        self.assertEqual(code, 1)
        self.assertNotIn("SYNTHETIC_PRIVATE_DATA", stdout + stderr)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.output.parent.exists())

    def test_changed_protocol_after_fitting_prevents_publication(self):
        def factory():
            self.protocol.write_bytes(self.protocol.read_bytes() + b"\n")
            return RecordingSameModelRunner()
        with self.loader_patches(), patch.object(cli, "SklearnV3SameModelRunner", side_effect=factory):
            self.assertEqual(self.invoke()[0], 1)
        self.assertFalse(self.output.exists())
