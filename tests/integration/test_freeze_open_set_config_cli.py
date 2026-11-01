from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from face_recognition.presentation.cli import freeze_open_set_config as cli
from tests.open_set_config_fakes import synthetic_development_report, synthetic_report_bytes


class FreezeOpenSetConfigCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="synthetic_freeze_cli_")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.source = self.directory / "development.json"
        self.source.write_bytes(synthetic_report_bytes())
        self.output = self.directory / "models/open_set_config_v1.json"

    def invoke(self, args=None):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            result = cli.main(["--development-report", str(self.source), "--output", str(self.output)] if args is None else args)
        return result, stdout.getvalue(), stderr.getvalue()

    def test_only_aggregate_report_opened_and_summary_contains_no_private_metadata(self):
        before = self.source.read_bytes(), self.source.stat().st_mtime_ns
        original, opened = Path.open, []

        def guarded_open(path, *args, **kwargs):
            self.assertEqual(path, self.source)
            self.assertNotEqual(path.suffix, ".jsonl")
            opened.append(path)
            return original(path, *args, **kwargs)

        with patch.object(Path, "open", guarded_open), \
             patch.object(Path, "glob", side_effect=AssertionError("No dataset discovery allowed")), \
             patch.object(Path, "rglob", side_effect=AssertionError("No dataset discovery allowed")):
            status, stdout, stderr = self.invoke()
        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(opened, [self.source])
        self.assertEqual((self.source.read_bytes(), self.source.stat().st_mtime_ns), before)
        for expected in ("CONFIGURATION FROZEN", "ANOVA k=448", "mean_5nn", "window: 7", "not numerically calibrated",
                         "known_test accessed: no", "unknown_holdout accessed: no"):
            self.assertIn(expected, stdout)
        for forbidden in ("synthetic_known", "synthetic_cal", "synthetic_hold", str(self.directory), "0.841531"):
            self.assertNotIn(forbidden, stdout)
        self.assertIsNone(json.loads(self.output.read_text(encoding="utf-8"))["final_numeric_threshold"])

    def test_report_mismatch_exits_nonzero_without_output(self):
        report = synthetic_development_report()
        report["model"]["k"] = 512
        self.source.write_text(json.dumps(report), encoding="utf-8")
        status, stdout, stderr = self.invoke()
        self.assertEqual(status, 1)
        self.assertEqual(stdout, "")
        self.assertIn("expected ANOVA-448", stderr)
        self.assertFalse(self.output.exists())

    def test_overwrite_refused_before_reading_source(self):
        self.assertEqual(self.invoke()[0], 0)
        before = self.output.read_bytes()
        with patch.object(cli, "read_aggregate_development_report") as read:
            status, stdout, stderr = self.invoke()
        self.assertEqual(status, 1)
        self.assertEqual(stdout, "")
        self.assertIn("already exists", stderr)
        read.assert_not_called()
        self.assertTrue(before == self.output.read_bytes())

    def test_errors_are_sanitized_and_missing_source_does_not_write_output(self):
        with patch.object(cli, "read_aggregate_development_report", side_effect=RuntimeError("SYNTHETIC_PRIVATE_MARKER")):
            status, stdout, stderr = self.invoke()
        self.assertEqual(status, 1)
        self.assertEqual(stdout, "")
        self.assertNotIn("SYNTHETIC_PRIVATE_MARKER", stderr)
        status, _, _ = self.invoke(["--development-report", str(self.directory / "missing.json"), "--output", str(self.output)])
        self.assertEqual(status, 1)
        self.assertFalse(self.output.exists())

    def test_usage_interrupt_and_no_unfreeze_option(self):
        for args in ([], ["--unfreeze"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                cli.main(args)
            self.assertEqual(error.exception.code, 2)
        with patch.object(cli, "read_aggregate_development_report", side_effect=KeyboardInterrupt):
            self.assertEqual(self.invoke()[0], 1)

    def test_command_import_needs_no_numpy_or_sklearn(self):
        result = subprocess.run([sys.executable, "-c",
            "import sys; import face_recognition.presentation.cli.freeze_open_set_config; "
            "assert 'numpy' not in sys.modules; assert 'sklearn' not in sys.modules"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
