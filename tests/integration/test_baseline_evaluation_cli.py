from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from face_recognition.domain.models.identity_split import IdentityPartition
from face_recognition.infrastructure.persistence.feature_metadata_reader import FeatureMetadataReader
from face_recognition.presentation.cli import evaluate_baseline_models as cli
from tests.baseline_fakes import baseline_fixture


class BaselineEvaluationCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="synthetic_baseline_cli_")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.processed, self.root, self.manifest, self.split = baseline_fixture(self.directory)
        self.output = self.directory / "reports/baseline.json"

    def args(self):
        return ["--processed-root", str(self.processed), "--split-manifest", str(self.split), "--report", str(self.output)]

    def invoke(self, args=None):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = cli.main(self.args() if args is None else args)
        return status, stdout.getvalue(), stderr.getvalue()

    def test_end_to_end_evaluation_opens_no_protected_payloads_and_leaves_sources_unchanged(self):
        protected = {self.processed / sample.source_relative_path for sample in self.manifest.samples
                     if sample.partition is not IdentityPartition.KNOWN_TRAIN_POOL}
        for path in protected:
            path.write_text("Protected synthetic payload must not be opened", encoding="utf-8")
        before = {path: (path.read_bytes(), path.stat().st_mtime_ns)
                  for path in self.processed.rglob("*") if path.is_file()}
        original_open = Path.open
        opened = []

        def guarded_open(path, *args, **kwargs):
            self.assertNotIn(path, protected)
            opened.append(path)
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", guarded_open), \
             patch.object(FeatureMetadataReader, "iter_metadata", side_effect=AssertionError("Full source scan forbidden")):
            status, stdout, stderr = self.invoke()
        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        self.assertIn("BASELINE IDENTITY EVALUATION", stdout)
        self.assertIn("Features: 9; known identities: 2", stdout)
        for name in ("KNN", "SVM", "Random Forest", "Fold A", "Fold B", "macro precision", "macro recall", "macro F1"):
            self.assertIn(name, stdout)
        self.assertNotIn("known_alpha", stdout)
        self.assertNotIn(str(self.directory), stdout)
        self.assertTrue(before == {path: (path.read_bytes(), path.stat().st_mtime_ns)
                                  for path in self.processed.rglob("*") if path.is_file()}, "Sources must remain unchanged")
        report = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(len(report["classifiers"]), 3)
        self.assertEqual([len(row["folds"]) for row in report["classifiers"]], [2, 2, 2])
        self.assertTrue(any(path.suffix == ".jsonl" for path in opened))

    def test_relative_paths_are_supported(self):
        previous = Path.cwd()
        try:
            os.chdir(self.directory)
            status, _, stderr = self.invoke(["--processed-root", "processed", "--split-manifest", "split.json", "--report", "relative.json"])
        finally:
            os.chdir(previous)
        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        self.assertTrue((self.directory / "relative.json").is_file())

    def test_existing_report_refused_before_loading_or_fitting(self):
        self.output.parent.mkdir()
        self.output.write_text("Existing completed aggregate report", encoding="utf-8")
        before = self.output.read_bytes()
        with patch.object(cli, "DevelopmentDatasetLoader") as loader:
            status, stdout, stderr = self.invoke()
        self.assertEqual(status, 1)
        self.assertEqual(stdout, "")
        self.assertIn("already exists", stderr)
        loader.assert_not_called()
        self.assertTrue(self.output.read_bytes() == before)

    def test_bad_features_and_backend_exception_messages_are_sanitized(self):
        path = self.root / "known_alpha/visit_a/center.jsonl"
        path.write_text('{"features": [987654.321', encoding="utf-8")
        status, stdout, stderr = self.invoke()
        self.assertEqual(status, 1)
        self.assertEqual(stdout, "")
        self.assertNotIn("987654", stderr)
        self.assertFalse(self.output.exists())
        with patch.object(cli, "DevelopmentDatasetLoader", side_effect=RuntimeError("SYNTHETIC_PRIVATE_PAYLOAD")):
            status, _, stderr = self.invoke()
        self.assertEqual(status, 1)
        self.assertNotIn("SYNTHETIC_PRIVATE_PAYLOAD", stderr)

    def test_usage_errors_return_two(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            cli.main([])
        self.assertEqual(caught.exception.code, 2)
