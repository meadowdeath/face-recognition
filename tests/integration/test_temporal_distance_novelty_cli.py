from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from face_recognition.domain.models.identity_split import IdentityPartition
from face_recognition.infrastructure.persistence.feature_metadata_reader import FeatureMetadataReader
from face_recognition.presentation.cli import evaluate_temporal_distance_novelty as cli
from tests.unknown_rejection_fakes import rejection_fixture


class TemporalDistanceNoveltyCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="synthetic_temporal_distance_cli_")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.processed, self.source, self.partition, self.source_path, self.partition_path = rejection_fixture(self.directory, samples=8)
        self.output = self.directory / "reports/temporal_distance.json"

    def args(self):
        return ["--processed-root", str(self.processed), "--identity-split", str(self.source_path),
                "--unknown-split", str(self.partition_path), "--report", str(self.output)]

    def invoke(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = cli.main(self.args())
        return status, stdout.getvalue(), stderr.getvalue()

    def test_protected_payloads_never_opened_no_source_changes_and_only_aggregate_output(self):
        protected = {self.processed / r.source_relative_path for r in self.source.samples
                     if r.partition is IdentityPartition.KNOWN_TEST or r.person_id in self.partition.specification.holdout_ids}
        for path in protected:
            path.write_text("SYNTHETIC_PROTECTED_PAYLOAD_MUST_NOT_BE_OPENED", encoding="utf-8")
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.processed.rglob("*") if p.is_file()}
        original_open, opened = Path.open, set()

        def guarded_open(path, *args, **kwargs):
            self.assertNotIn(path, protected)
            opened.add(path)
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", guarded_open), \
             patch.object(FeatureMetadataReader, "iter_metadata", side_effect=AssertionError("Full source scan forbidden")):
            status, stdout, stderr = self.invoke()
        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        for expected in ("nearest_1", "mean_5nn", "window=1", "window=3", "window=5", "window=7", "FAR<=5%", "discarded=",
                         "known_test biometric payloads accessed: no", "unknown_holdout biometric payloads accessed: no"):
            self.assertIn(expected, stdout)
        for forbidden in ("mean_3nn", "known_a", "unknown_cal", str(self.directory), "array(", "nearest_labels"):
            self.assertNotIn(forbidden, stdout)
        self.assertTrue(before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.processed.rglob("*") if p.is_file()})
        allowed = {self.processed / r.source_relative_path for r in self.source.samples
                   if r.partition is IdentityPartition.KNOWN_TRAIN_POOL or r.person_id in self.partition.specification.calibration_ids}
        self.assertEqual({p for p in opened if p.suffix == ".jsonl"}, allowed)
        self.assertEqual(len(json.loads(self.output.read_text(encoding="utf-8"))["results"]), 8)

    def test_completed_output_refused_before_loading(self):
        self.output.parent.mkdir()
        self.output.write_text("Existing completed report", encoding="utf-8")
        with patch.object(cli, "DevelopmentDatasetLoader") as loader:
            status, stdout, stderr = self.invoke()
        self.assertEqual(status, 1)
        self.assertEqual(stdout, "")
        self.assertIn("already exists", stderr)
        loader.assert_not_called()

    def test_internal_errors_sanitized(self):
        with patch.object(cli, "DevelopmentDatasetLoader", side_effect=ValueError("PRIVATE_SYNTHETIC_VALUES")):
            status, stdout, stderr = self.invoke()
        self.assertEqual(status, 1)
        self.assertEqual(stdout, "")
        self.assertNotIn("PRIVATE_SYNTHETIC_VALUES", stderr)
        self.assertFalse(self.output.exists())

    def test_usage_and_interrupt_exit_codes(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            cli.main([])
        self.assertEqual(error.exception.code, 2)
        with patch.object(cli, "DevelopmentDatasetLoader", side_effect=KeyboardInterrupt):
            status, stdout, stderr = self.invoke()
        self.assertEqual(status, 1)
        self.assertEqual(stdout, "")
        self.assertIn("interrupted", stderr)
