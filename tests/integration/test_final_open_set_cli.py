from contextlib import ExitStack, redirect_stdout, redirect_stderr
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from face_recognition.domain.models.identity_split import IdentityPartition
from face_recognition.infrastructure.persistence.model_repository import model_fingerprints
from face_recognition.presentation.cli import evaluate_final_open_set as cli
from tests.final_evaluation_fakes import final_fixture


class FinalOpenSetCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="synthetic_final_cli_")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.bundle = final_fixture(self.directory)
        self.fixture, self.development, self.model_dir = self.bundle
        self.output = self.directory / "reports/final.json"

    def invoke(self):
        processed, _, _, source_path, partition_path, config_path = self.fixture
        args = ["--processed-root", str(processed), "--identity-split", str(source_path), "--unknown-split", str(partition_path),
                "--model-dir", str(self.model_dir), "--frozen-config", str(config_path),
                "--development-report", str(self.development), "--report", str(self.output)]
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = cli.main(args)
        return status, stdout.getvalue(), stderr.getvalue()

    def test_end_to_end_final_only_no_fitting_or_calibration_and_all_sources_immutable(self):
        processed, source, partition, _, _, _ = self.fixture
        forbidden = {processed / s.source_relative_path for s in source.samples
                     if s.partition is IdentityPartition.KNOWN_TRAIN_POOL or s.person_id in partition.specification.calibration_ids}
        for path in forbidden:
            path.write_text("SYNTHETIC_DEVELOPMENT_PAYLOAD_FORBIDDEN_DURING_FINAL", encoding="utf-8")
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in processed.rglob("*") if p.is_file()}
        before_model = model_fingerprints(self.model_dir)
        original, opened = Path.open, set()

        def guard(path, *args, **kwargs):
            self.assertNotIn(path, forbidden)
            opened.add(path)
            return original(path, *args, **kwargs)

        with ExitStack() as stack:
            stack.enter_context(patch.object(Path, "open", guard))
            for target in ("sklearn.feature_selection.SelectKBest.fit", "sklearn.preprocessing.StandardScaler.fit", "sklearn.svm.SVC.fit",
                           "face_recognition.infrastructure.features.known_distances.KnownDistanceScorer.fit",
                           "face_recognition.application.rejection_metrics.calibrate_threshold",
                           "face_recognition.application.rejection_score_analysis.calibrate_threshold"):
                stack.enter_context(patch(target, side_effect=AssertionError("Final stage must not fit or calibrate")))
            status, stdout, stderr = self.invoke()
        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        for expected in ("CLOSED-SET KNOWN TEST", "FINAL OPEN-SET TEMPORAL TEST", "CALIBRATION PERFORMANCE (NOT GENERALIZATION)",
                         "known_test biometric payloads accessed: yes", "unknown_holdout biometric payloads accessed: yes",
                         "modify this model version"):
            self.assertIn(expected, stdout)
        for forbidden_text in ('"known_a"', "array(", str(self.directory), "support_vectors"):
            self.assertNotIn(forbidden_text, stdout)
        allowed = {processed / s.source_relative_path for s in source.samples
                   if s.partition is IdentityPartition.KNOWN_TEST or s.person_id in partition.specification.holdout_ids}
        self.assertEqual({p for p in opened if p.suffix == ".jsonl"}, allowed)
        self.assertTrue(before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in processed.rglob("*") if p.is_file()})
        self.assertEqual(model_fingerprints(self.model_dir), before_model)

    def test_existing_final_report_refused_before_holdout_access(self):
        self.output.parent.mkdir()
        self.output.write_text("Existing completed final report", encoding="utf-8")
        with patch.object(cli, "FinalHoldoutLoader") as loader:
            status, stdout, _ = self.invoke()
        self.assertEqual(status, 1)
        self.assertEqual(stdout, "")
        loader.assert_not_called()

    def test_corrupted_model_or_changed_development_report_refused_before_holdouts(self):
        for path in (self.model_dir / "threshold.json", self.development):
            before = path.read_bytes()
            try:
                path.write_bytes(before + b"\n")
                with patch.object(cli, "FinalHoldoutLoader") as loader:
                    status, stdout, _ = self.invoke()
                self.assertEqual(status, 1)
                self.assertEqual(stdout, "")
                loader.assert_not_called()
            finally:
                path.write_bytes(before)

    def test_no_override_flags_and_errors_sanitized(self):
        for args in ([], ["--force"], ["--threshold", "0"], ["--window-size", "3"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                cli.main(args)
            self.assertEqual(error.exception.code, 2)
        with patch.object(cli, "read_frozen_configuration", side_effect=ValueError("SYNTHETIC_PRIVATE_MARKER")):
            status, stdout, stderr = self.invoke()
        self.assertEqual(status, 1)
        self.assertEqual(stdout, "")
        self.assertNotIn("SYNTHETIC_PRIVATE_MARKER", stderr)
        with patch.object(cli, "read_frozen_configuration", side_effect=KeyboardInterrupt):
            status, stdout, stderr = self.invoke()
        self.assertEqual(status, 1)
        self.assertEqual(stdout, "")
        self.assertIn("interrupted", stderr)
        self.assertFalse(self.output.exists())
