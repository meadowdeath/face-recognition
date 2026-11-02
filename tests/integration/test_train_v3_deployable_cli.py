from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import warnings

from face_recognition.infrastructure.persistence.v3_deployable_model import load_v3_deployable_bundle
from face_recognition.presentation.cli import train_v3_deployable_model as cli
from tests.v3_deployable_fakes import deployable_fixture


class TrainV3DeployableCLITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory(prefix="synthetic_v3_deployable_cli_")
        cls.folder = Path(cls.temporary.name)
        cls.data, cls.config, cls.provenance, cls.root, cls.config_path, *_ = deployable_fixture(cls.folder)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def invoke(self, output, extra=()):
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.main(["--processed-root", str(self.root), "--frozen-config", str(self.config_path), "--output-dir", str(output), *extra])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_success_aggregate_calibration_only_and_loadable_bundle(self):
        output = self.folder / "models" / "deployable"
        before = self.config_path.read_bytes(), self.config_path.stat().st_mtime_ns
        with patch("face_recognition.infrastructure.persistence.v3_deployable_dataset.load_v3_payloads", return_value=self.data), warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning); warnings.simplefilter("ignore", RuntimeWarning)
            code, stdout, stderr = self.invoke(output)
        self.assertEqual((code, stderr), (0, ""))
        self.assertIn("CALIBRATION METRICS", stdout)
        self.assertIn("not independent final test", stdout)
        self.assertIn("fit generation=1", stdout)
        self.assertIn("No new V3 evaluation", stdout)
        self.assertNotIn(str(self.folder), stdout)
        self.assertNotIn("known_00", stdout)
        self.assertEqual(before, (self.config_path.read_bytes(), self.config_path.stat().st_mtime_ns))
        self.assertIsNotNone(load_v3_deployable_bundle(output))

    def test_existing_directory_refused_before_input_loading(self):
        output = self.folder / "existing"
        output.mkdir()
        (output / "evidence").write_text("immutable", encoding="utf-8")
        with patch.object(cli, "V3DeployableInputs") as inputs:
            self.assertEqual(self.invoke(output)[0], 1)
            inputs.assert_not_called()
        self.assertEqual((output / "evidence").read_text(encoding="utf-8"), "immutable")

    def test_actual_loader_rejects_invalid_synthetic_jsonl_before_fit(self):
        # Metadata-only fixture has deliberately invalid numeric payload markers.
        output = self.folder / "bad_payload"
        with patch.object(cli, "SklearnV3DeployableRunner") as runner:
            self.assertEqual(self.invoke(output)[0], 1)
            runner.assert_not_called()
        self.assertFalse(output.exists())

    def test_training_failure_publishes_nothing_and_hides_private_exception(self):
        output = self.folder / "failed"
        with patch("face_recognition.infrastructure.persistence.v3_deployable_dataset.load_v3_payloads", return_value=self.data), \
                patch.object(cli.TrainV3DeployableModel, "train", side_effect=RuntimeError("SYNTHETIC_PRIVATE_VALUE")):
            code, stdout, stderr = self.invoke(output)
        self.assertEqual(code, 1)
        self.assertNotIn("SYNTHETIC_PRIVATE_VALUE", stdout + stderr)
        self.assertFalse(output.exists())
        self.assertEqual(tuple(self.folder.glob("failed.*.tmp")), ())

    def test_no_hyperparameter_role_or_force_overrides(self):
        for option in ("--force", "--C", "--window-size", "--fit-session", "--novelty-method"):
            with self.subTest(option=option), self.assertRaises(SystemExit) as raised:
                self.invoke(self.folder / "unused", (option,))
            self.assertEqual(raised.exception.code, 2)
