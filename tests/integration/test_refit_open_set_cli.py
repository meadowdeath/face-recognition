from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from face_recognition.domain.models.identity_split import IdentityPartition
from face_recognition.infrastructure.persistence.feature_metadata_reader import FeatureMetadataReader
from face_recognition.presentation.cli import refit_open_set_model as cli
from tests.open_set_refit_fakes import refit_fixture


class RefitOpenSetCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="synthetic_refit_cli_")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.fixture = refit_fixture(self.directory)
        self.processed, self.source, self.partition, self.source_path, self.partition_path, self.config_path = self.fixture
        self.output = self.directory / "models/fitted_v1"

    def invoke(self):
        args = ["--processed-root", str(self.processed), "--identity-split", str(self.source_path),
                "--unknown-split", str(self.partition_path), "--frozen-config", str(self.config_path), "--output-dir", str(self.output)]
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = cli.main(args)
        return status, stdout.getvalue(), stderr.getvalue()

    def test_holdout_files_never_opened_sources_unchanged_and_aggregate_output(self):
        protected = {self.processed / s.source_relative_path for s in self.source.samples
                     if s.partition is IdentityPartition.KNOWN_TEST or s.person_id in self.partition.specification.holdout_ids}
        for path in protected:
            path.write_text("SYNTHETIC_PROTECTED_DO_NOT_OPEN", encoding="utf-8")
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.processed.rglob("*") if p.is_file()}
        original, opened = Path.open, set()

        def guarded_open(path, *args, **kwargs):
            self.assertNotIn(path, protected)
            opened.add(path)
            return original(path, *args, **kwargs)

        with patch.object(Path, "open", guarded_open), \
             patch.object(FeatureMetadataReader, "iter_metadata", side_effect=AssertionError("No full feature scan")):
            status, stdout, stderr = self.invoke()
        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        for expected in ("known_test biometric payloads accessed: no", "unknown_holdout biometric payloads accessed: no",
                         "Known training samples=96", "windows=12", "windows=4", "Threshold=", "NOT been performed"):
            self.assertIn(expected, stdout)
        for forbidden in ("known_a", "unknown_cal", "array(", str(self.directory), "training accuracy"):
            self.assertNotIn(forbidden, stdout)
        allowed = {self.processed / s.source_relative_path for s in self.source.samples
                   if s.partition is IdentityPartition.KNOWN_TRAIN_POOL or s.person_id in self.partition.specification.calibration_ids}
        self.assertEqual({p for p in opened if p.suffix == ".jsonl"}, allowed)
        self.assertTrue(before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.processed.rglob("*") if p.is_file()})
        self.assertTrue((self.output / "manifest.json").is_file())

    def test_existing_output_refused_before_data_loading(self):
        self.output.mkdir(parents=True)
        with patch.object(cli, "DevelopmentDatasetLoader") as loader:
            status, stdout, _ = self.invoke()
        self.assertEqual(status, 1)
        self.assertEqual(stdout, "")
        loader.assert_not_called()

    def test_inconsistent_frozen_config_refused_before_dataset_loading(self):
        data = json.loads(self.config_path.read_text(encoding="utf-8"))
        data["architecture"]["svm_parameters"]["C"] = 99.
        self.config_path.write_text(json.dumps(data), encoding="utf-8")
        with patch.object(cli, "DevelopmentDatasetLoader") as loader:
            status, stdout, _ = self.invoke()
        self.assertEqual(status, 1)
        self.assertEqual(stdout, "")
        loader.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_unexpected_error_sanitized_usage_and_override_rejected(self):
        with patch.object(cli, "read_frozen_configuration", side_effect=ValueError("PRIVATE_SYNTHETIC_VALUES")):
            status, stdout, stderr = self.invoke()
        self.assertEqual(status, 1)
        self.assertEqual(stdout, "")
        self.assertNotIn("PRIVATE_SYNTHETIC_VALUES", stderr)
        for args in ([], ["--force"], ["--C", "2"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                cli.main(args)
            self.assertEqual(error.exception.code, 2)
