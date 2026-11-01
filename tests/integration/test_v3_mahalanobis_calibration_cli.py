from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from face_recognition.infrastructure.persistence.v3_cv_protocol_repository import write_v3_cv_protocol
from face_recognition.presentation.cli import evaluate_v3_mahalanobis_calibration as cli
from tests.v3_mahalanobis_fakes import RecordingSameModelRunner, write_selection_report
from tests.v3_novelty_fakes import novelty_data


class V3MahalanobisCalibrationCLITests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="synthetic_v3_calibration_cli_")
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.data, self.processed, self.spec_path = novelty_data(self.folder)
        self.protocol_path = self.folder / "protocol.json"
        write_v3_cv_protocol(self.data.protocol, self.protocol_path, (self.processed, self.spec_path))
        self.selection_path = write_selection_report(self.data, self.folder / "selection.json")
        self.output = self.folder / "reports" / "calibration.json"

    def invoke(self, output=None):
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = cli.main(["--processed-root", str(self.processed), "--protocol", str(self.protocol_path),
                "--selection-report", str(self.selection_path), "--report", str(output or self.output)])
        return status, stdout.getvalue(), stderr.getvalue()

    def test_success_prints_aggregate_refinement_and_keeps_source_report_immutable(self):
        before = self.selection_path.read_bytes(), self.protocol_path.read_bytes()
        with patch.object(cli, "load_v3_payloads", return_value=self.data), \
                patch.object(cli, "SklearnV3SameModelRunner", return_value=RecordingSameModelRunner()):
            status, stdout, stderr = self.invoke()
        self.assertEqual((status, stderr), (0, ""))
        self.assertIn("threshold_transport=none_same_model_calibration", stdout)
        self.assertIn("no refit after calibration", stdout)
        self.assertIn("Prior outer results motivated this refinement", stdout)
        self.assertNotIn("known_00", stdout)
        self.assertNotIn("novel_00", stdout)
        self.assertNotIn(str(self.folder), stdout)
        self.assertEqual(before, (self.selection_path.read_bytes(), self.protocol_path.read_bytes()))
        raw = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(raw["configuration"]["novelty_method"], "mahalanobis_distance")

    def test_overwrite_existing_output_or_original_report_refused_before_inputs(self):
        self.output.parent.mkdir()
        self.output.write_text("completed synthetic output", encoding="utf-8")
        for target in (self.output, self.selection_path):
            before = target.read_bytes()
            with patch.object(cli, "read_v3_cv_protocol", side_effect=AssertionError("Must refuse first")), \
                    patch.object(cli, "load_v3_payloads", side_effect=AssertionError("Must not open payloads")):
                self.assertEqual(self.invoke(target)[0], 1)
            self.assertEqual(target.read_bytes(), before)

    def test_bad_protocol_or_selection_is_rejected_before_payloads(self):
        original = self.protocol_path.read_bytes()
        self.protocol_path.write_text("{}", encoding="utf-8")
        with patch.object(cli, "load_v3_payloads") as loader:
            self.assertEqual(self.invoke()[0], 1)
            loader.assert_not_called()
        self.protocol_path.write_bytes(original)
        source = json.loads(self.selection_path.read_text(encoding="utf-8"))
        source["protocol_sha256"] = "0" * 64
        self.selection_path.write_text(json.dumps(source), encoding="utf-8")
        with patch.object(cli, "load_v3_payloads") as loader:
            self.assertEqual(self.invoke()[0], 1)
            loader.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_actual_strict_loader_rejects_malformed_synthetic_payload_before_fitting(self):
        # The metadata fixture intentionally contains invalid JSONL. Use the
        # real retained loader, not a substituted data object for this test.
        with patch.object(cli, "SklearnV3SameModelRunner") as runner:
            self.assertEqual(self.invoke()[0], 1)
            runner.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_fit_or_immutability_failure_publishes_no_partial_report(self):
        runner = RecordingSameModelRunner()
        runner.fit = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("SYNTHETIC_PRIVATE_STATE"))
        with patch.object(cli, "load_v3_payloads", return_value=self.data), patch.object(cli, "SklearnV3SameModelRunner", return_value=runner):
            status, stdout, stderr = self.invoke()
        self.assertEqual(status, 1)
        self.assertNotIn("SYNTHETIC_PRIVATE_STATE", stdout + stderr)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.output.parent.exists())

    def test_changed_selection_report_during_execution_refuses_publication(self):
        runner = RecordingSameModelRunner()
        original = runner.fit
        def fit(*args, **kwargs):
            self.selection_path.write_bytes(self.selection_path.read_bytes() + b"\n")
            return original(*args, **kwargs)
        runner.fit = fit
        with patch.object(cli, "load_v3_payloads", return_value=self.data), patch.object(cli, "SklearnV3SameModelRunner", return_value=runner):
            self.assertEqual(self.invoke()[0], 1)
        self.assertFalse(self.output.exists())
