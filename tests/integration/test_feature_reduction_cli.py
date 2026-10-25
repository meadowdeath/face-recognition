from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from face_recognition.domain.models.identity_split import IdentityPartition
from face_recognition.presentation.cli import evaluate_feature_reduction as cli
from tests.baseline_fakes import baseline_fixture


class FeatureReductionCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="synthetic_reduction_cli_")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.processed, _, self.manifest, self.split = baseline_fixture(self.directory)
        self.output = self.directory / "reports/reduction.json"

    def args(self):
        return ["--processed-root", str(self.processed), "--split-manifest", str(self.split), "--report", str(self.output),
                "--anova-k", "2", "--anova-k", "4", "--pca-variance", ".95"]

    def invoke(self, args=None):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = cli.main(self.args() if args is None else args)
        return status, stdout.getvalue(), stderr.getvalue()

    def test_end_to_end_known_train_only_with_aggregate_output_and_unchanged_sources(self):
        protected = {self.processed / row.source_relative_path for row in self.manifest.samples
                     if row.partition is not IdentityPartition.KNOWN_TRAIN_POOL}
        for path in protected:
            path.write_text("UNREADABLE_SYNTHETIC_PROTECTED_PAYLOAD", encoding="utf-8")
        before = {path: path.read_bytes() for path in self.processed.rglob("*") if path.is_file()}
        original = Path.open

        def guarded_open(path, *args, **kwargs):
            self.assertNotIn(path, protected)
            return original(path, *args, **kwargs)

        with patch.object(Path, "open", guarded_open):
            status, stdout, stderr = self.invoke()
        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        self.assertIn("FEATURE REDUCTION EVALUATION", stdout)
        self.assertIn("Ranked development configurations", stdout)
        for expected in ("anova_k_2", "anova_k_4", "pca_0.95", "KNN", "SVM", "Random Forest", "Fold A", "Fold B"):
            self.assertIn(expected, stdout)
        self.assertNotIn("known_alpha", stdout)
        self.assertNotIn("selected_feature_indices", stdout)
        self.assertNotIn(str(self.directory), stdout)
        self.assertTrue(before == {path: path.read_bytes() for path in self.processed.rglob("*") if path.is_file()})
        self.assertEqual(len(json.loads(self.output.read_text(encoding="utf-8"))["results"]), 12)

    def test_existing_report_refused_before_loading(self):
        self.output.parent.mkdir()
        self.output.write_text("Existing completed report", encoding="utf-8")
        with patch.object(cli, "DevelopmentDatasetLoader") as loader:
            status, stdout, stderr = self.invoke()
        self.assertEqual(status, 1)
        self.assertEqual(stdout, "")
        self.assertIn("already exists", stderr)
        loader.assert_not_called()

    def test_invalid_configuration_is_explicit_and_error_values_are_sanitized(self):
        status, _, stderr = self.invoke([*self.args(), "--anova-k", "0"])
        self.assertEqual(status, 1)
        self.assertIn("Invalid reduction configuration", stderr)
        with patch.object(cli, "DevelopmentDatasetLoader", side_effect=ValueError("SYNTHETIC_PRIVATE_FEATURE_VALUE")):
            status, stdout, stderr = self.invoke()
        self.assertEqual(status, 1)
        self.assertEqual(stdout, "")
        self.assertNotIn("SYNTHETIC_PRIVATE_FEATURE_VALUE", stderr)
        self.assertFalse(self.output.exists())

    def test_usage_error_is_two(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            cli.main([])
        self.assertEqual(caught.exception.code, 2)
