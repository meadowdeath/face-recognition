from contextlib import chdir, redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
import unittest

from face_recognition.domain.models.processed_sample import ProcessingSummary, REPRESENTATIONS
from face_recognition.presentation.cli import process_dataset as cli
from tests.processed_dataset_fakes import write_synthetic_session


class ProcessDatasetCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory(prefix="synthetic_processing_cli_")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.raw = self.directory / "raw"
        self.output = self.directory / "processed"
        self.raw.mkdir()
        self.session = write_synthetic_session(self.raw)

    def invoke(self, raw: Path | None = None, output: Path | None = None) -> tuple[int, str, str]:
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = cli.main(["--raw-root", str(raw or self.raw), "--output-root", str(output or self.output)])
        return status, stdout.getvalue(), stderr.getvalue()

    @staticmethod
    def snapshot(directory: Path) -> dict:
        return {path.relative_to(directory).as_posix():
                (path.read_bytes() if path.is_file() else None, path.stat().st_mtime_ns)
                for path in (directory, *directory.rglob("*"))}

    def test_success_with_relative_paths_resolves_output_and_creates_both_representations(self) -> None:
        before = self.snapshot(self.raw)
        with chdir(self.directory):
            status, stdout, stderr = self.invoke(Path("raw"), Path("processed"))
        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        self.assertIn(f"Output directory: {self.output.resolve()}", stdout)
        for representation in REPRESENTATIONS:
            directory = self.output / representation
            self.assertTrue((directory / "processing_manifest.json").is_file())
            self.assertTrue((directory / "subject_test/visit_test/center.jsonl").is_file())
            self.assertTrue((directory / "subject_test/visit_test/translation_left.jsonl").is_file())
        self.assertTrue(self.snapshot(self.raw) == before, "CLI processing must not rewrite raw files or directory entries")

    def test_success_with_absolute_paths_prints_aggregate_summary_only(self) -> None:
        write_synthetic_session(self.raw, "another_subject", "visit_a")
        write_synthetic_session(self.raw, "another_subject", "visit_b")
        status, stdout, stderr = self.invoke()
        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        for expected in ("Processing completed.", "Participants: 2", "Sessions: 3",
                         "Conditions/files per representation: 6", "Total condition files: 12",
                         "Samples: 12", "normalized_landmarks_v1", "all_xyz_v1"):
            self.assertIn(expected, stdout)
        for identifier in ("subject_test", "another_subject", "visit_test", "visit_a", "visit_b"):
            self.assertNotIn(identifier, stdout)
        self.assertNotIn("normalized_landmarks\":", stdout)
        self.assertNotIn("features\":", stdout)

    def test_missing_raw_root_returns_failure_without_creating_output(self) -> None:
        absent = self.directory / "missing_raw"
        status, stdout, stderr = self.invoke(raw=absent)
        self.assertEqual(status, 1)
        self.assertEqual(stdout, "")
        self.assertIn("existing directory", stderr)
        self.assertFalse(absent.exists())
        self.assertFalse(self.output.exists())

    def test_source_output_overlap_is_rejected_without_raw_changes(self) -> None:
        before = self.snapshot(self.raw)
        for output in (self.raw, self.raw / "derived", self.directory):
            with self.subTest(case=output.name):
                status, stdout, stderr = self.invoke(output=output)
                self.assertEqual(status, 1)
                self.assertEqual(stdout, "")
                self.assertIn("overlap", stderr)
        self.assertTrue(self.snapshot(self.raw) == before, "Invalid output paths must not alter raw data")

    def test_repository_local_output_is_rejected_by_existing_policy(self) -> None:
        simulated_repo = self.directory / "simulated_repo"
        with patch("face_recognition.infrastructure.persistence.processed_dataset_repository.PROJECT_ROOT", simulated_repo):
            status, stdout, stderr = self.invoke(output=simulated_repo / "derived")
        self.assertEqual(status, 1)
        self.assertEqual(stdout, "")
        self.assertIn("outside the Git repository", stderr)
        self.assertFalse(simulated_repo.exists())

    def test_existing_completed_output_is_refused_without_modification(self) -> None:
        self.assertEqual(self.invoke()[0], 0)
        before = self.snapshot(self.output)
        status, stdout, stderr = self.invoke()
        self.assertEqual(status, 1)
        self.assertEqual(stdout, "")
        self.assertIn("already exists", stderr)
        self.assertTrue(self.snapshot(self.output) == before, "Existing output must never be overwritten")

    def test_existing_empty_output_is_refused(self) -> None:
        self.output.mkdir()
        status, _, stderr = self.invoke()
        self.assertEqual(status, 1)
        self.assertIn("already exists", stderr)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_normalization_failure_preserves_safe_context_and_raw_source(self) -> None:
        path = self.session / "center.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        data = json.loads(lines[1])
        for index in (33, 133, 362, 263):
            data["landmarks"][index] = [0, 0, 0]
        lines[1] = json.dumps(data)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        before = self.snapshot(self.raw)
        status, stdout, stderr = self.invoke()
        self.assertEqual(status, 1)
        self.assertEqual(stdout, "")
        for context in ("normalization_failed", "subject_test", "visit_test", "center", "sample_index=1"):
            self.assertIn(context, stderr)
        self.assertNotIn("Traceback", stderr)
        self.assertNotIn(lines[1], stderr)
        self.assertFalse(self.output.exists())
        self.assertTrue(self.snapshot(self.raw) == before, "Processing failure must leave the raw dataset unchanged")

    def test_malformed_sample_contents_are_not_printed_to_stderr(self) -> None:
        marker = "SYNTHETIC_PRIVATE_SAMPLE_TEXT"
        (self.session / "center.jsonl").write_text(marker, encoding="utf-8")
        status, stdout, stderr = self.invoke()
        self.assertEqual(status, 1)
        self.assertEqual(stdout, "")
        self.assertIn("source_validation_failed", stderr)
        self.assertIn("sample_index=0", stderr)
        self.assertNotIn(marker, stderr)

    def test_unexpected_exception_values_are_not_printed(self) -> None:
        marker = "SYNTHETIC_PRIVATE_FEATURE_VALUES"
        with patch.object(cli, "DatasetProcessor") as processor:
            processor.return_value.process.side_effect = RuntimeError(marker)
            status, stdout, stderr = self.invoke()
        self.assertEqual(status, 1)
        self.assertEqual(stdout, "")
        self.assertIn("unexpected internal or filesystem error", stderr)
        self.assertNotIn(marker, stderr)

    def test_output_and_manifests_do_not_embed_personal_absolute_source_paths(self) -> None:
        status, _, _ = self.invoke()
        self.assertEqual(status, 0)

        def strings(value):
            if isinstance(value, str):
                yield value
            elif isinstance(value, dict):
                for item in value.values():
                    yield from strings(item)
            elif isinstance(value, list):
                for item in value:
                    yield from strings(item)

        for path in self.output.rglob("*"):
            if not path.is_file():
                continue
            for line in ([path.read_text(encoding="utf-8")] if path.suffix == ".json"
                         else path.read_text(encoding="utf-8").splitlines()):
                for value in strings(json.loads(line)):
                    self.assertFalse(Path(value).is_absolute(), "Persisted metadata must use relative provenance")
                    self.assertNotIn(str(self.directory), value)

    def test_cli_only_composes_and_delegates_to_existing_components(self) -> None:
        reader, repository, processor, extractor = Mock(), Mock(), Mock(), Mock()
        reader.root = self.raw.resolve()
        repository.output_root = self.output.resolve()
        processor.process.return_value = ProcessingSummary(1, 1, 2, 4)
        with patch.object(cli, "RawDatasetReader", return_value=reader) as make_reader, \
             patch.object(cli, "FilesystemProcessedDatasetRepository", return_value=repository) as make_repository, \
             patch.object(cli, "LandmarkFeatureExtractor", return_value=extractor) as make_extractor, \
             patch.object(cli, "DatasetProcessor", return_value=processor) as make_processor:
            with chdir(self.directory):
                status, _, _ = self.invoke(Path("raw"), Path("processed"))
        self.assertEqual(status, 0)
        make_reader.assert_called_once_with(Path("raw"))
        make_repository.assert_called_once_with(Path("processed"), reader.root)
        make_extractor.assert_called_once_with()
        make_processor.assert_called_once_with(cli.LandmarkNormalizer, extractor)
        processor.process.assert_called_once_with(reader, repository)
        repository.write_sample.assert_not_called()
        extractor.extract.assert_not_called()

    def test_missing_required_arguments_use_argparse_usage_exit_code(self) -> None:
        for args in ([], ["--raw-root", str(self.raw)], ["--output-root", str(self.output)]):
            with self.subTest(argument_count=len(args)), redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    cli.main(args)
                self.assertEqual(caught.exception.code, 2)

    def test_keyboard_interrupt_returns_documented_failure_code(self) -> None:
        with patch.object(cli, "DatasetProcessor") as processor:
            processor.return_value.process.side_effect = KeyboardInterrupt
            status, stdout, stderr = self.invoke()
        self.assertEqual(status, 1)
        self.assertEqual(stdout, "")
        self.assertIn("interrupted", stderr)


if __name__ == "__main__":
    unittest.main()
