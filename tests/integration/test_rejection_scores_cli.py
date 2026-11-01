from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from face_recognition.domain.models.identity_split import IdentityPartition
from face_recognition.infrastructure.persistence.feature_metadata_reader import FeatureMetadataReader
from face_recognition.presentation.cli import evaluate_rejection_scores as cli
from tests.unknown_rejection_fakes import rejection_fixture


class RejectionScoresCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="synthetic_score_cli_")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.processed, self.source, self.partition, self.source_path, self.partition_path = rejection_fixture(self.directory)
        self.output = self.directory / "reports/scores.json"

    def args(self):
        return ["--processed-root", str(self.processed), "--identity-split", str(self.source_path),
                "--unknown-split", str(self.partition_path), "--report", str(self.output)]

    def invoke(self, args=None):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = cli.main(self.args() if args is None else args)
        return status, stdout.getvalue(), stderr.getvalue()

    def test_protected_files_never_opened_sources_unchanged_and_output_aggregate_only(self):
        protected = {self.processed / row.source_relative_path for row in self.source.samples
                     if row.partition is IdentityPartition.KNOWN_TEST or row.person_id in self.partition.specification.holdout_ids}
        for path in protected:
            path.write_text("SYNTHETIC_PROTECTED_PAYLOAD_MUST_NOT_BE_OPENED", encoding="utf-8")
        before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in self.processed.rglob("*") if path.is_file()}
        original = Path.open
        opened = set()

        def guarded_open(path, *args, **kwargs):
            self.assertNotIn(path, protected)
            opened.add(path)
            return original(path, *args, **kwargs)

        with patch.object(Path, "open", guarded_open), \
             patch.object(FeatureMetadataReader, "iter_metadata", side_effect=AssertionError("Full source scan forbidden")):
            status, stdout, stderr = self.invoke()
        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        for expected in ("svm_margin", "svm_top1", "centroid_distance", "normalized_centroid_distance", "FAR <= 5%",
                         "known_test biometric payloads accessed: no", "unknown_holdout biometric payloads accessed: no"):
            self.assertIn(expected, stdout)
        self.assertNotIn("known_a", stdout)
        self.assertNotIn(str(self.directory), stdout)
        self.assertNotIn("known_confidence", stdout)
        self.assertTrue(before == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in self.processed.rglob("*") if path.is_file()})
        allowed = {self.processed / row.source_relative_path for row in self.source.samples
                   if row.partition is IdentityPartition.KNOWN_TRAIN_POOL or row.person_id in self.partition.specification.calibration_ids}
        self.assertEqual({path for path in opened if path.suffix == ".jsonl"}, allowed)
        self.assertEqual(len(json.loads(self.output.read_text(encoding="utf-8"))["results"]), 8)

    def test_existing_report_refused_before_loading(self):
        self.output.parent.mkdir()
        self.output.write_text("Existing completed report", encoding="utf-8")
        with patch.object(cli, "DevelopmentDatasetLoader") as loader:
            status, stdout, stderr = self.invoke()
        self.assertEqual(status, 1)
        self.assertEqual(stdout, "")
        self.assertIn("already exists", stderr)
        loader.assert_not_called()

    def test_errors_do_not_expose_private_values(self):
        with patch.object(cli, "DevelopmentDatasetLoader", side_effect=ValueError("PRIVATE_SYNTHETIC_COORDINATES")):
            status, stdout, stderr = self.invoke()
        self.assertEqual(status, 1)
        self.assertEqual(stdout, "")
        self.assertNotIn("PRIVATE_SYNTHETIC_COORDINATES", stderr)
        self.assertFalse(self.output.exists())

    def test_missing_arguments_return_two(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            cli.main([])
        self.assertEqual(caught.exception.code, 2)
