from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from face_recognition.domain.models.identity_split import IdentityPartition
from face_recognition.infrastructure.persistence.feature_metadata_reader import FeatureMetadataReader
from face_recognition.presentation.cli import tune_anova_svm as cli
from tests.anova_svm_fakes import tuning_fixture


class AnovaSvmTuningCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="synthetic_tuning_cli_")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.processed, _, self.manifest, self.split = tuning_fixture(self.directory)
        self.output = self.directory / "reports/tuning.json"

    def args(self):
        return ["--processed-root", str(self.processed), "--split-manifest", str(self.split), "--report", str(self.output)]

    def invoke(self, args=None):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            result = cli.main(self.args() if args is None else args)
        return result, stdout.getvalue(), stderr.getvalue()

    def test_real_pipeline_grid_runs_only_on_synthetic_training_pool_and_preserves_sources(self):
        protected = {self.processed / row.source_relative_path for row in self.manifest.samples
                     if row.partition is not IdentityPartition.KNOWN_TRAIN_POOL}
        for path in protected:
            path.write_text("PROTECTED_SYNTHETIC_DATA_MUST_NOT_BE_OPENED", encoding="utf-8")
        before = {path: path.read_bytes() for path in self.processed.rglob("*") if path.is_file()}
        original = Path.open

        def guarded_open(path, *args, **kwargs):
            self.assertNotIn(path, protected)
            return original(path, *args, **kwargs)

        with patch.object(Path, "open", guarded_open), \
             patch.object(FeatureMetadataReader, "iter_metadata", side_effect=AssertionError("Full-pool scan prohibited")):
            status, stdout, stderr = self.invoke()
        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        self.assertIn("Configurations evaluated: 60", stdout)
        self.assertIn("Selected development candidate:", stdout)
        self.assertIn("support vectors:", stdout)
        self.assertIn("feature overlap=", stdout)
        self.assertNotIn("known_alpha", stdout)
        self.assertNotIn("confusion_matrix", stdout)
        self.assertNotIn("fold_a_indices", stdout)
        self.assertNotIn(str(self.directory), stdout)
        self.assertTrue(before == {path: path.read_bytes() for path in self.processed.rglob("*") if path.is_file()})
        self.assertEqual(len(json.loads(self.output.read_text(encoding="utf-8"))["results"]), 60)

    def test_report_overwrite_refused_before_loading(self):
        self.output.parent.mkdir()
        self.output.write_text("Completed report", encoding="utf-8")
        with patch.object(cli, "DevelopmentDatasetLoader") as loader:
            status, stdout, stderr = self.invoke()
        self.assertEqual(status, 1)
        self.assertEqual(stdout, "")
        self.assertIn("already exists", stderr)
        loader.assert_not_called()

    def test_source_errors_and_unexpected_values_are_sanitized(self):
        path = self.processed / "all_xyz_v1/known_alpha/visit_a/center.jsonl"
        path.write_text('{"features": [987654.321', encoding="utf-8")
        status, stdout, stderr = self.invoke()
        self.assertEqual(status, 1)
        self.assertEqual(stdout, "")
        self.assertNotIn("987654", stderr)
        self.assertFalse(self.output.exists())
        with patch.object(cli, "DevelopmentDatasetLoader", side_effect=ValueError("PRIVATE_SYNTHETIC_VALUES")):
            status, _, stderr = self.invoke()
        self.assertEqual(status, 1)
        self.assertNotIn("PRIVATE_SYNTHETIC_VALUES", stderr)

    def test_missing_arguments_return_two_and_no_grid_overrides_are_exposed(self):
        for args in ([], [*self.args(), "--k", "1024"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                cli.main(args)
            self.assertEqual(caught.exception.code, 2)
