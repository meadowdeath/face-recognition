from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from face_recognition.domain.models.identity_split import IdentityPartition
from face_recognition.infrastructure.persistence.feature_metadata_reader import FeatureMetadataReader
from face_recognition.presentation.cli import calibrate_unknown_rejection as calibrate_cli
from face_recognition.presentation.cli import create_unknown_rejection_split as split_cli
from tests.unknown_rejection_fakes import rejection_fixture


class UnknownRejectionCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="synthetic_rejection_cli_")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.processed, self.source, self.partition, self.source_path, self.partition_path = rejection_fixture(self.directory)
        self.output = self.directory / "reports/calibration.json"

    def invoke(self, main, args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            result = main(args)
        return result, stdout.getvalue(), stderr.getvalue()

    def args(self):
        return ["--processed-root", str(self.processed), "--identity-split", str(self.source_path),
                "--unknown-split", str(self.partition_path), "--report", str(self.output)]

    def test_create_and_verify_metadata_only_counts_deterministic_no_overwrite(self):
        target = self.directory / "splits/created.json"
        args = ["--identity-split", str(self.source_path), "--output", str(target), "--calibration-id", "unknown_cal",
                "--holdout-id", "unknown_hold_b", "--holdout-id", "unknown_hold_a"]
        original = Path.open

        def metadata_only(path, *args, **kwargs):
            self.assertNotEqual(path.suffix, ".jsonl")
            return original(path, *args, **kwargs)

        with patch.object(Path, "open", metadata_only):
            status, stdout, stderr = self.invoke(split_cli.main, args)
            verified, summary, _ = self.invoke(split_cli.main, ["--identity-split", str(self.source_path), "--verify", str(target)])
        self.assertEqual((status, verified), (0, 0))
        self.assertEqual(stderr, "")
        self.assertIn("identities=1; sessions=2; condition files=4; samples=12", stdout)
        self.assertIn("Unknown rejection split verified", summary)
        self.assertNotIn("unknown_cal", stdout)
        self.assertEqual(self.invoke(split_cli.main, args)[0], 1)

    def test_calibration_never_opens_known_test_or_unknown_holdout_and_keeps_sources_unchanged(self):
        protected = {self.processed / row.source_relative_path for row in self.source.samples
                     if row.partition is IdentityPartition.KNOWN_TEST or row.person_id in self.partition.specification.holdout_ids}
        for path in protected:
            path.write_text("UNREADABLE_PROTECTED_SYNTHETIC_FEATURES", encoding="utf-8")
        before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in self.processed.rglob("*") if path.is_file()}
        original = Path.open
        opened = set()

        def guarded_open(path, *args, **kwargs):
            self.assertNotIn(path, protected)
            opened.add(path)
            return original(path, *args, **kwargs)

        with patch.object(Path, "open", guarded_open), \
             patch.object(FeatureMetadataReader, "iter_metadata", side_effect=AssertionError("Full-pool scan forbidden")):
            status, stdout, stderr = self.invoke(calibrate_cli.main, self.args())
        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        self.assertIn("Holdout biometric payloads accessed: no", stdout)
        self.assertIn("ANOVA-512", stdout)
        self.assertIn("ANOVA-448", stdout)
        self.assertIn("Selected open-set development candidate", stdout)
        self.assertNotIn("known_a", stdout)
        self.assertNotIn("known_confidence", stdout)
        self.assertTrue(before == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in self.processed.rglob("*") if path.is_file()})
        allowed = {self.processed / row.source_relative_path for row in self.source.samples
                   if row.partition is IdentityPartition.KNOWN_TRAIN_POOL or row.person_id in self.partition.specification.calibration_ids}
        self.assertEqual({path for path in opened if path.suffix == ".jsonl"}, allowed)
        self.assertEqual(len(json.loads(self.output.read_text(encoding="utf-8"))["results"]), 2)

    def test_existing_report_is_refused_before_any_payload_loading(self):
        self.output.parent.mkdir()
        self.output.write_text("completed report", encoding="utf-8")
        with patch.object(calibrate_cli, "DevelopmentDatasetLoader") as loader:
            status, stdout, stderr = self.invoke(calibrate_cli.main, self.args())
        self.assertEqual(status, 1)
        self.assertEqual(stdout, "")
        self.assertIn("already exists", stderr)
        loader.assert_not_called()

    def test_bad_calibration_payload_error_is_sanitized(self):
        path = self.processed / "all_xyz_v1/unknown_cal/u1/center.jsonl"
        path.write_text('{"features": [987654.321', encoding="utf-8")
        status, stdout, stderr = self.invoke(calibrate_cli.main, self.args())
        self.assertEqual(status, 1)
        self.assertEqual(stdout, "")
        self.assertNotIn("987654", stderr)
        self.assertFalse(self.output.exists())

    def test_usage_errors_for_missing_or_conflicting_roles(self):
        for main, args in ((calibrate_cli.main, []), (split_cli.main, []),
                           (split_cli.main, ["--identity-split", str(self.source_path), "--output", str(self.output)]),
                           (split_cli.main, ["--identity-split", str(self.source_path), "--verify", str(self.partition_path),
                                             "--calibration-id", "unknown_cal"])):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                main(args)
            self.assertEqual(caught.exception.code, 2)
