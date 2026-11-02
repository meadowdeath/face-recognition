from contextlib import ExitStack, redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from face_recognition.presentation.cli import create_v3_cv_protocol as cli
from tests.v3_protocol_fakes import v3_catalog_fixture


class V3ProtocolCLITests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="synthetic_v3_cli_")
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.processed, self.spec_path, _ = v3_catalog_fixture(self.folder, known_count=2)
        self.output = self.folder / "splits" / "protocol.json"

    def invoke(self, *extra):
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = cli.main(["--processed-root", str(self.processed), "--experiment-spec", str(self.spec_path),
                               "--output", str(self.output), *extra])
        return status, stdout.getvalue(), stderr.getvalue()

    def test_create_and_validate_are_aggregate_only_with_no_fitting_or_search(self):
        original = Path.open
        def deny_payload(path, *args, **kwargs):
            if path.suffix == ".jsonl":
                raise AssertionError("No feature payload access")
            return original(path, *args, **kwargs)
        with ExitStack() as stack:
            stack.enter_context(patch.object(Path, "open", deny_payload))
            for name in ("sklearn.feature_selection.SelectKBest.fit", "sklearn.preprocessing.StandardScaler.fit",
                         "sklearn.svm.SVC.fit", "face_recognition.application.rejection_metrics.calibrate_threshold"):
                stack.enter_context(patch(name, side_effect=AssertionError("No model execution")))
            status, stdout, stderr = self.invoke()
            self.assertEqual((status, stderr), (0, ""))
            before = self.output.read_bytes(), self.output.stat().st_mtime_ns
            status, validated, stderr = self.invoke("--validate-only")
            self.assertEqual((status, stderr), (0, ""))
            self.assertIn("VALIDATED", validated)
            self.assertEqual(before, (self.output.read_bytes(), self.output.stat().st_mtime_ns))
        for text in ("KNOWN identities: 2", "UNKNOWN identities: 8", "Outer folds: 4", "Inner splits per outer fold: 18",
                     "Biometric payloads accessed: no"):
            self.assertIn(text, stdout)
        for marker in ("known_00", "novel_00", "SYNTHETIC_PAYLOAD_DO_NOT_READ", str(self.folder)):
            self.assertNotIn(marker, stdout + validated)

    def test_overwrite_refused_before_catalog_inspection(self):
        self.assertEqual(self.invoke()[0], 0)
        before = self.output.read_bytes()
        with patch.object(cli, "read_v3_catalog", side_effect=AssertionError("Must refuse before catalog access")):
            self.assertEqual(self.invoke()[0], 1)
        self.assertEqual(self.output.read_bytes(), before)

    def test_validate_only_rejects_changed_source_metadata_without_writing(self):
        self.assertEqual(self.invoke()[0], 0)
        before = self.output.read_bytes(), self.output.stat().st_mtime_ns
        header = self.processed / "all_xyz_v1" / "processing_manifest.json"
        data = json.loads(header.read_text(encoding="utf-8"))
        data["normalization_note"] = "synthetic metadata change"
        header.write_text(json.dumps(data), encoding="utf-8")
        self.assertEqual(self.invoke("--validate-only")[0], 1)
        self.assertEqual(before, (self.output.read_bytes(), self.output.stat().st_mtime_ns))

    def test_validate_only_requires_existing_output(self):
        self.assertEqual(self.invoke("--validate-only")[0], 1)
        self.assertFalse(self.output.exists())
