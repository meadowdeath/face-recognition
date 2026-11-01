from contextlib import chdir, redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

from face_recognition.presentation.cli import evaluate_normalization as cli
from tests.normalization_quality_fakes import process_quality_fixture, snapshot, write_quality_session


class EvaluateNormalizationCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory(prefix="synthetic_quality_cli_")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.raw = self.directory / "raw"
        self.raw.mkdir()
        self.processed = self.directory / "processed"
        self.session = write_quality_session(self.raw, transforms={"translation_left": {"translation": (0.1, 0, 0)}})
        process_quality_fixture(self.raw, self.processed)

    def invoke(self, args=None):
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = cli.main(args or ["--raw-root", str(self.raw), "--processed-root", str(self.processed)])
        return status, stdout.getvalue(), stderr.getvalue()

    def test_relative_paths_print_only_aggregate_statistics_and_write_optional_report(self) -> None:
        before_raw, before_processed = snapshot(self.raw), snapshot(self.processed)
        with chdir(self.directory):
            status, stdout, stderr = self.invoke(["--raw-root", "raw", "--processed-root", "processed",
                                                  "--report", "reports/quality.json"])
        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        for text in ("Targeted invariance conditions", "Pose conditions", "Raw(ref)", "RawStd(w)", "NormStd",
                     "Reduction%", "sample pairs: 26", "no pass/fail thresholds"):
            self.assertIn(text, stdout)
        self.assertNotIn("synthetic_subject", stdout)
        self.assertNotIn("synthetic_visit", stdout)
        report = json.loads((self.directory / "reports/quality.json").read_text(encoding="utf-8"))
        self.assertEqual(report["evaluated_sample_pairs"], 26)
        self.assertTrue(snapshot(self.raw) == before_raw, "CLI evaluation must preserve raw input")
        self.assertTrue(snapshot(self.processed) == before_processed, "CLI evaluation must preserve processed input")

    def test_absolute_paths_work_without_a_report_file(self) -> None:
        status, _, stderr = self.invoke()
        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(list(self.directory.glob("*.json")), [])

    def test_missing_roots_fail_without_creating_them(self) -> None:
        missing = self.directory / "missing"
        status, stdout, stderr = self.invoke(["--raw-root", str(missing), "--processed-root", str(self.processed)])
        self.assertEqual(status, 1)
        self.assertEqual(stdout, "")
        self.assertIn("Evaluation failed", stderr)
        self.assertFalse(missing.exists())

    def test_report_inside_input_tree_is_refused(self) -> None:
        target = self.raw / "quality.json"
        status, stdout, stderr = self.invoke(["--raw-root", str(self.raw), "--processed-root", str(self.processed),
                                               "--report", str(target)])
        self.assertEqual(status, 1)
        self.assertEqual(stdout, "")
        self.assertIn("report location", stderr)
        self.assertFalse(target.exists())

    def test_invalid_payload_and_unexpected_error_contents_are_never_exposed(self) -> None:
        marker = "SYNTHETIC_PRIVATE_BIOMETRIC_PAYLOAD"
        (self.session / "center.jsonl").write_text(marker, encoding="utf-8")
        status, stdout, stderr = self.invoke()
        self.assertEqual(status, 1)
        self.assertEqual(stdout, "")
        self.assertNotIn(marker, stderr)
        self.assertNotIn("Traceback", stderr)
        with patch.object(cli.NormalizationQualityEvaluator, "evaluate", side_effect=RuntimeError(marker)):
            status, stdout, stderr = self.invoke()
        self.assertEqual(status, 1)
        self.assertNotIn(marker, stdout + stderr)

    def test_usage_errors_return_argparse_code_two(self) -> None:
        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit) as caught:
                cli.main([])
        self.assertEqual(caught.exception.code, 2)
