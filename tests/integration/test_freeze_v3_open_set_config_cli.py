from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from face_recognition.infrastructure.persistence.v3_open_set_config import read_frozen_v3_configuration
from face_recognition.presentation.cli import freeze_v3_open_set_config as cli
from tests.v3_frozen_config_fakes import frozen_evidence_fixture


class FreezeV3ConfigurationCLITests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="synthetic_v3_freeze_cli_")
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.paths = frozen_evidence_fixture(self.folder)
        self.output = self.folder / "models" / "config.json"

    def invoke(self, output=None, extra=()):
        args = [arg for kind, path in self.paths.items() for arg in ("--" + kind.replace("_", "-"), str(path))]
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.main([*args, "--output", str(output or self.output), *extra])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_success_aggregate_output_deterministic_configuration_and_evidence_immutable(self):
        before = {k: (p.read_bytes(), p.stat().st_mtime_ns) for k, p in self.paths.items()}
        code, stdout, stderr = self.invoke()
        self.assertEqual((code, stderr), (0, ""))
        self.assertIn("all_xyz_pose_v2", stdout)
        self.assertIn("none_same_model_calibration", stdout)
        self.assertIn("not calibrated", stdout)
        self.assertIn("not defined", stdout)
        self.assertIn("not independent final generalization", stdout)
        self.assertNotIn(str(self.folder), stdout)
        self.assertEqual(read_frozen_v3_configuration(self.output).pipeline.selected_features, 448)
        self.assertEqual(self.invoke(self.folder / "second.json")[0], 0)
        self.assertEqual(self.output.read_bytes(), (self.folder / "second.json").read_bytes())
        self.assertEqual(before, {k: (p.read_bytes(), p.stat().st_mtime_ns) for k, p in self.paths.items()})

    def test_existing_output_or_evidence_alias_refused_before_loading(self):
        self.output.parent.mkdir()
        self.output.write_text("immutable", encoding="utf-8")
        for output in (self.output, self.paths["pose_report"]):
            before = output.read_bytes()
            with patch("face_recognition.infrastructure.persistence.v3_open_set_config.load_v3_freeze_evidence") as loader:
                self.assertEqual(self.invoke(output)[0], 1)
                loader.assert_not_called()
            self.assertEqual(output.read_bytes(), before)

    def test_bad_evidence_returns_nonzero_and_no_partial_configuration(self):
        self.paths["pose_report"].write_text("{}", encoding="utf-8")
        code, stdout, stderr = self.invoke()
        self.assertEqual(code, 1)
        self.assertNotIn(str(self.folder), stdout + stderr)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.output.parent.exists())

    def test_no_model_parameter_overrides_or_force_option(self):
        for option in ("--force", "--window-size", "--C", "--novelty-method"):
            with self.subTest(option=option), self.assertRaises(SystemExit) as caught:
                self.invoke(extra=(option,))
            self.assertEqual(caught.exception.code, 2)

    def test_failure_does_not_expose_private_exception_values(self):
        with patch.object(cli, "freeze_v3_configuration_file", side_effect=ValueError("SYNTHETIC_PRIVATE_VALUE")):
            code, stdout, stderr = self.invoke()
        self.assertEqual(code, 1)
        self.assertNotIn("SYNTHETIC_PRIVATE_VALUE", stdout + stderr)
        self.assertFalse(self.output.exists())
