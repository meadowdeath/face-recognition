from contextlib import chdir, redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

from face_recognition.application.create_identity_split import IdentitySplitPlanner
from face_recognition.domain.models.identity_split import IdentitySplitSpec
from face_recognition.infrastructure.persistence import identity_split_repository as persistence
from face_recognition.infrastructure.persistence.feature_metadata_reader import FeatureMetadataReader
from face_recognition.infrastructure.persistence.identity_split_repository import IdentitySplitPersistenceError, write_identity_split
from face_recognition.presentation.cli import create_identity_split as cli
from tests.identity_split_fakes import write_feature_dataset
from tests.normalization_quality_fakes import snapshot


class IdentitySplitCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory(prefix="synthetic_identity_cli_")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.processed = self.directory / "processed"
        self.output = self.directory / "splits/split.json"
        write_feature_dataset(self.processed, {"known_example": ("a", "b", "c"), "unknown_example": ("u", "v")})
        self.spec = IdentitySplitSpec(("known_example",), ("unknown_example",), ("a", "b"), ("c",))

    def arguments(self, processed=None, output=None):
        return ["--processed-root", str(processed or self.processed), "--output", str(output or self.output),
                "--known-id", "known_example", "--unknown-id", "unknown_example",
                "--train-session", "a", "--train-session", "b", "--known-test-session", "c"]

    def invoke(self, arguments=None):
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = cli.main(arguments or self.arguments())
        return status, stdout.getvalue(), stderr.getvalue()

    def test_success_with_relative_paths_prints_only_aggregate_counts(self) -> None:
        before = snapshot(self.processed)
        with chdir(self.directory):
            status, stdout, stderr = self.invoke(self.arguments(Path("processed"), Path("splits/split.json")))
        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        for text in ("Known identities: 1", "Unknown identities: 1", "Known training samples: 8", "Known test samples: 4",
                     "Unknown pool samples: 8", "Total samples: 20"):
            self.assertIn(text, stdout)
        self.assertNotIn("known_example", stdout)
        self.assertNotIn("unknown_example", stdout)
        self.assertTrue(snapshot(self.processed) == before, "Split generation must not modify processed data")
        data = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(data["completion_state"], "completed")
        self.assertEqual(data["total_sample_count"], 20)

    def test_manifest_serialization_is_deterministic_and_contains_no_absolute_paths_or_values(self) -> None:
        manifest = IdentitySplitPlanner().plan(FeatureMetadataReader(self.processed), self.spec)
        second = self.directory / "split_again.json"
        write_identity_split(manifest, self.output, self.processed)
        write_identity_split(manifest, second, self.processed)
        self.assertTrue(self.output.read_bytes() == second.read_bytes(), "Split serialization must be deterministic")
        data = json.loads(self.output.read_text(encoding="utf-8"))
        def check(value):
            if isinstance(value, dict):
                self.assertFalse(set(value) & {"features", "landmarks", "normalized_landmarks"})
                for child in value.values():
                    check(child)
            elif isinstance(value, list):
                for child in value:
                    check(child)
            elif isinstance(value, str):
                self.assertFalse(Path(value).is_absolute())
                self.assertNotIn(str(self.directory), value)
        check(data)

    def test_existing_completed_split_is_refused_without_modification(self) -> None:
        self.assertEqual(self.invoke()[0], 0)
        before = self.output.read_bytes()
        status, stdout, stderr = self.invoke()
        self.assertEqual(status, 1)
        self.assertEqual(stdout, "")
        self.assertIn("already exists", stderr)
        self.assertTrue(self.output.read_bytes() == before, "Existing split must not be overwritten")

    def test_invalid_roles_fail_before_source_discovery(self) -> None:
        with patch.object(cli, "FeatureMetadataReader") as source:
            status, _, stderr = self.invoke([*self.arguments(), "--known-id", "unknown_example"])
        self.assertEqual(status, 1)
        self.assertIn("both known and unknown", stderr)
        source.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_source_errors_and_unexpected_exception_contents_are_sanitized(self) -> None:
        path = self.processed / "all_xyz_v1/known_example/a/center.jsonl"
        marker = "SYNTHETIC_PRIVATE_FEATURE_PAYLOAD"
        path.write_text(marker, encoding="utf-8")
        status, stdout, stderr = self.invoke()
        self.assertEqual(status, 1)
        self.assertEqual(stdout, "")
        self.assertNotIn(marker, stderr)
        self.assertFalse(self.output.exists())
        with patch.object(cli, "FeatureMetadataReader", side_effect=RuntimeError(marker)):
            status, _, stderr = self.invoke()
        self.assertEqual(status, 1)
        self.assertNotIn(marker, stderr)

    def test_output_inside_source_or_repository_is_refused(self) -> None:
        manifest = IdentitySplitPlanner().plan(FeatureMetadataReader(self.processed), self.spec)
        with self.assertRaisesRegex(IdentitySplitPersistenceError, "outside the processed"):
            write_identity_split(manifest, self.processed / "split.json", self.processed)
        simulated_repo = self.directory / "repository"
        with patch.object(persistence, "PROJECT_ROOT", simulated_repo):
            with self.assertRaisesRegex(IdentitySplitPersistenceError, "outside the Git"):
                write_identity_split(manifest, simulated_repo / "split.json", self.processed)
        self.assertFalse(simulated_repo.exists())

    def test_atomic_publication_failure_does_not_leave_a_completed_split(self) -> None:
        with patch.object(persistence.os, "link", side_effect=OSError("test")):
            status, _, _ = self.invoke()
        self.assertEqual(status, 1)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.output.parent.glob("*.tmp")), [])

    def test_usage_and_identifier_errors_have_code_two(self) -> None:
        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit) as caught:
                cli.main([])
            self.assertEqual(caught.exception.code, 2)
            with self.assertRaises(SystemExit) as caught:
                cli.main([*self.arguments(), "--known-id", "../outside"])
            self.assertEqual(caught.exception.code, 2)
