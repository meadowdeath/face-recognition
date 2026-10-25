from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

from face_recognition.domain.models.dataset import CONDITIONS, CaptureProtocol, SessionManifest
from face_recognition.infrastructure.persistence.dataset_repository import (
    FilesystemDatasetRepository, manifest_to_dict, sample_to_json,
)
from face_recognition.infrastructure.persistence.raw_dataset_reader import RawDatasetReader, RawDatasetReadError
from tests.dataset_fakes import condition_samples, session_config


class RawDatasetReaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory(prefix="synthetic_raw_reader_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "raw"
        self.root.mkdir()
        self.reader = RawDatasetReader(self.root)

    def write_session(
        self, person: str = "subject_test", session: str = "visit_test",
        completed: tuple[str, ...] = CONDITIONS[:2], landmark_count: int = 3,
    ) -> tuple[Path, SessionManifest]:
        config = session_config(person_id=person, session_id=session, protocol=CaptureProtocol(5, 2, 250))
        manifest = SessionManifest(config, "2026-10-24T12:00:00+00:00",
                                   landmark_count if completed else None, completed)
        directory = self.root / person / session
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text(json.dumps(manifest_to_dict(manifest)), encoding="utf-8")
        # Deliberately create files in reverse order; discovery must not depend on it.
        for condition in reversed(completed):
            samples = condition_samples(config, condition, landmark_count)
            (directory / (condition + ".jsonl")).write_text(
                "".join(sample_to_json(sample) + "\n" for sample in samples), encoding="utf-8",
            )
        return directory, manifest

    def change_record(self, directory: Path, changes: dict, line_index: int = 0, condition: str = "center") -> None:
        path = directory / (condition + ".jsonl")
        lines = path.read_text(encoding="utf-8").splitlines()
        data = json.loads(lines[line_index])
        data.update(changes)
        lines[line_index] = json.dumps(data)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def assert_invalid_samples(self) -> None:
        with self.assertRaises(RawDatasetReadError):
            list(self.reader.iter_samples())

    def test_discovery_is_deterministic_for_participants_sessions_conditions_and_samples(self) -> None:
        expected_sessions = [("alpha_test", "visit_a"), ("alpha_test", "visit_z"), ("zeta_test", "visit_b")]
        for person, session in reversed(expected_sessions):
            self.write_session(person, session, CONDITIONS)
        sessions = self.reader.discover_sessions()
        self.assertEqual([(item.manifest.config.person_id, item.manifest.config.session_id) for item in sessions],
                         expected_sessions)
        keys = [(sample.person_id, sample.session_id, sample.condition, sample.sample_index)
                for sample in self.reader.iter_samples()]
        expected = [(person, session, condition, index) for person, session in expected_sessions
                    for condition in CONDITIONS for index in range(2)]
        self.assertEqual(keys, expected)
        self.assertEqual(keys, [(s.person_id, s.session_id, s.condition, s.sample_index)
                                for s in self.reader.iter_samples()])

    def test_valid_decoding_preserves_coordinates_order_and_metadata(self) -> None:
        directory, manifest = self.write_session(landmark_count=17)
        discovered = self.reader.discover_sessions()[0]
        actual = self.reader.read_condition(discovered, "center")
        expected = condition_samples(manifest.config, "center", landmark_count=17)
        self.assertTrue(actual == expected, "Decoded samples must exactly match the synthetic source")
        self.assertEqual(discovered.directory, directory)
        self.assertEqual(len(actual[0].landmarks.points), 17)

    def test_manifest_can_be_read_by_absolute_or_root_relative_session_path(self) -> None:
        directory, manifest = self.write_session()
        self.assertEqual(self.reader.read_manifest(directory), manifest)
        self.assertEqual(self.reader.read_manifest(Path("subject_test") / "visit_test"), manifest)
        self.assertEqual(self.reader.discover_sessions()[0].manifest, manifest)

    def test_root_relative_session_descriptor_is_resolved_under_the_reader_root(self) -> None:
        self.write_session()
        session = self.reader.discover_sessions()[0]
        relative = replace(session, directory=Path("subject_test") / "visit_test")
        self.assertEqual(len(self.reader.read_condition(relative, "center")), 2)
        self.assertEqual(len(tuple(self.reader.iter_samples(relative))), 4)

    def test_single_session_iteration_preserves_session_boundary(self) -> None:
        self.write_session("subject_a", "visit_a")
        self.write_session("subject_b", "visit_b")
        session = self.reader.discover_sessions()[1]
        samples = tuple(self.reader.iter_samples(session))
        self.assertEqual(len(samples), 4)
        self.assertTrue(all(sample.person_id == "subject_b" and sample.session_id == "visit_b" for sample in samples))

    def test_arbitrary_landmark_counts_use_the_manifest_without_fixed_478(self) -> None:
        for index, count in enumerate((1, 5, 478, 500)):
            self.write_session(session=f"visit_{index}", landmark_count=count)
        for session in self.reader.discover_sessions():
            count = session.manifest.expected_landmark_count
            self.assertTrue(all(len(sample.landmarks.points) == count for sample in self.reader.iter_samples(session)))

    def test_manifest_person_id_must_match_its_directory(self) -> None:
        directory, manifest = self.write_session()
        changed = replace(manifest, config=replace(manifest.config, person_id="different_subject"))
        (directory / "manifest.json").write_text(json.dumps(manifest_to_dict(changed)), encoding="utf-8")
        with self.assertRaisesRegex(RawDatasetReadError, "person_id"):
            self.reader.discover_sessions()

    def test_manifest_session_id_must_match_its_directory(self) -> None:
        directory, manifest = self.write_session()
        changed = replace(manifest, config=replace(manifest.config, session_id="different_visit"))
        (directory / "manifest.json").write_text(json.dumps(manifest_to_dict(changed)), encoding="utf-8")
        with self.assertRaisesRegex(RawDatasetReadError, "session_id"):
            self.reader.discover_sessions()

    def test_record_person_id_must_match_filesystem_and_manifest(self) -> None:
        directory, _ = self.write_session()
        self.change_record(directory, {"person_id": "another_subject"})
        self.assert_invalid_samples()

    def test_record_session_id_must_match_filesystem_and_manifest(self) -> None:
        directory, _ = self.write_session()
        self.change_record(directory, {"session_id": "another_visit"})
        self.assert_invalid_samples()

    def test_record_condition_must_match_its_filename(self) -> None:
        directory, _ = self.write_session()
        self.change_record(directory, {"condition": "translation_left"})
        self.assert_invalid_samples()

    def test_malformed_jsonl_fails_explicitly_without_exposing_source_text(self) -> None:
        directory, _ = self.write_session()
        secret = "SYNTHETIC_SOURCE_MARKER_NOT_FOR_LOGGING"
        (directory / "center.jsonl").write_text(secret, encoding="utf-8")
        with self.assertRaises(RawDatasetReadError) as caught:
            list(self.reader.iter_samples())
        self.assertNotIn(secret, str(caught.exception))
        self.assertTrue(caught.exception.__suppress_context__)

    def test_decoding_error_exposes_expected_file_index_without_source_values(self) -> None:
        directory, _ = self.write_session()
        path = directory / "center.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        marker = "SYNTHETIC_INVALID_SAMPLE_NOT_FOR_LOGGING"
        lines[1] = marker
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with self.assertRaises(RawDatasetReadError) as caught:
            list(self.reader.iter_samples())
        self.assertEqual(caught.exception.sample_index, 1)
        self.assertIn("sample_index=1", str(caught.exception))
        self.assertNotIn(marker, str(caught.exception))

    def test_malformed_or_nonobject_manifest_is_rejected(self) -> None:
        directory, _ = self.write_session()
        for content in ("not JSON", "[]", "null", '{"unknown":NaN}'):
            with self.subTest(case=content[:8]):
                (directory / "manifest.json").write_text(content, encoding="utf-8")
                with self.assertRaisesRegex(RawDatasetReadError, "manifest JSON/schema"):
                    self.reader.discover_sessions()

    def test_unknown_manifest_schema_is_rejected(self) -> None:
        directory, manifest = self.write_session()
        data = manifest_to_dict(manifest)
        data["schema_version"] = 999
        (directory / "manifest.json").write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaises(RawDatasetReadError):
            self.reader.discover_sessions()

    def test_invalid_sample_structures_and_nonfinite_coordinates_are_rejected(self) -> None:
        directory, _ = self.write_session()
        for changes in ({"landmarks": []}, {"landmarks": [[0, 1]]}, {"landmarks": [[None, 0, 0]]},
                        {"landmarks": [[float("nan"), 0, 0]]}, {"landmarks": [[float("inf"), 0, 0]]},
                        {"landmarks": [[0, 0, float("-inf")]]}, {"sample_index": -1}, {"schema_version": 999}):
            with self.subTest(field=next(iter(changes))):
                # Restore the baseline line before each isolated mutation.
                sample = condition_samples(session_config(person_id="subject_test", session_id="visit_test",
                                                          protocol=CaptureProtocol(5, 2, 250)), landmark_count=3)[0]
                lines = (directory / "center.jsonl").read_text(encoding="utf-8").splitlines()
                lines[0] = sample_to_json(sample)
                (directory / "center.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
                self.change_record(directory, changes)
                self.assert_invalid_samples()

    def test_invalid_or_manifest_inconsistent_dimensions_are_rejected(self) -> None:
        directory, _ = self.write_session()
        for width, height in ((0, 270), (480, -1), (True, 270), (480.0, 270), (640, 270), (480, 480)):
            self.change_record(directory, {"inference_width": width, "inference_height": height})
            self.assert_invalid_samples()

    def test_duplicate_missing_or_reversed_sample_indices_are_rejected_not_sorted(self) -> None:
        directory, manifest = self.write_session()
        baseline = [sample_to_json(sample) for sample in condition_samples(manifest.config, landmark_count=3)]
        path = directory / "center.jsonl"
        for indexes in ((0, 0), (0, 2), (1, 0)):
            lines = []
            for source, index in zip(baseline, indexes):
                data = json.loads(source)
                data["sample_index"] = index
                lines.append(json.dumps(data))
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            self.assert_invalid_samples()

    def test_sample_count_and_timestamp_spacing_follow_manifest_protocol(self) -> None:
        directory, manifest = self.write_session()
        samples = condition_samples(manifest.config, landmark_count=3)
        path = directory / "center.jsonl"
        path.write_text(sample_to_json(samples[0]) + "\n", encoding="utf-8")
        self.assert_invalid_samples()
        for timestamp in (samples[0].detector_timestamp_ms, samples[0].detector_timestamp_ms + 249):
            changed = replace(samples[1], detector_timestamp_ms=timestamp)
            path.write_text(sample_to_json(samples[0]) + "\n" + sample_to_json(changed) + "\n", encoding="utf-8")
            self.assert_invalid_samples()

    def test_landmark_count_must_match_manifest(self) -> None:
        directory, manifest = self.write_session()
        replacement = condition_samples(manifest.config, landmark_count=5)[0]
        lines = (directory / "center.jsonl").read_text(encoding="utf-8").splitlines()
        lines[0] = sample_to_json(replacement)
        (directory / "center.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.assert_invalid_samples()

    def test_missing_manifest_and_declared_condition_files_are_errors(self) -> None:
        directory, _ = self.write_session()
        (directory / "center.jsonl").unlink()
        with self.assertRaisesRegex(RawDatasetReadError, "Missing expected completed"):
            self.reader.discover_sessions()
        (directory / "manifest.json").unlink()
        with self.assertRaisesRegex(RawDatasetReadError, "Missing expected manifest"):
            self.reader.discover_sessions()

    def test_empty_root_is_valid_but_missing_root_is_not_created(self) -> None:
        (self.root / ".gitkeep").write_text("", encoding="utf-8")
        self.assertEqual(self.reader.discover_sessions(), ())
        absent = self.root / "missing"
        with self.assertRaises(RawDatasetReadError):
            RawDatasetReader(absent)
        self.assertFalse(absent.exists())

    def test_incomplete_sessions_expose_only_declared_conditions_and_leave_temps_alone(self) -> None:
        directory, _ = self.write_session(completed=CONDITIONS[:1])
        (directory / "translation_left.jsonl.tmp").write_text("incomplete synthetic temporary data", encoding="utf-8")
        (directory / "manifest.json.tmp").write_text("incomplete synthetic manifest", encoding="utf-8")
        session = self.reader.discover_sessions()[0]
        self.assertEqual(len(tuple(self.reader.iter_samples(session))), 2)
        with self.assertRaisesRegex(RawDatasetReadError, "not declared completed"):
            self.reader.read_condition(session, "translation_left")

    def test_uncommitted_final_file_is_not_recovered_or_silently_skipped(self) -> None:
        directory, manifest = self.write_session(completed=CONDITIONS[:1])
        (directory / "translation_left.jsonl").write_text(
            "".join(sample_to_json(sample) + "\n" for sample in condition_samples(manifest.config, "translation_left", 3)),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(RawDatasetReadError, "Uncommitted"):
            self.reader.discover_sessions()

    def test_changed_manifest_after_discovery_is_rejected(self) -> None:
        directory, manifest = self.write_session()
        session = self.reader.discover_sessions()[0]
        changed = replace(manifest, config=replace(manifest.config, orientation="rotate180"))
        (directory / "manifest.json").write_text(json.dumps(manifest_to_dict(changed)), encoding="utf-8")
        with self.assertRaisesRegex(RawDatasetReadError, "changed since discovery"):
            self.reader.read_condition(session, "center")

    def test_symlinked_or_escaping_session_paths_are_rejected(self) -> None:
        directory, _ = self.write_session()
        with patch.object(Path, "is_symlink", lambda path: path == directory):
            with self.assertRaisesRegex(RawDatasetReadError, "without symlinks"):
                self.reader.discover_sessions()
        with self.assertRaisesRegex(RawDatasetReadError, "inside the root"):
            self.reader.read_manifest(Path("..") / "outside" / "visit")

    def test_unexpected_hierarchy_entries_are_not_silently_skipped(self) -> None:
        (self.root / "unexpected.jsonl").write_text("synthetic invalid hierarchy", encoding="utf-8")
        with self.assertRaises(RawDatasetReadError):
            self.reader.discover_sessions()

    def test_reading_never_rewrites_files_or_calls_capture_mutations(self) -> None:
        directory, _ = self.write_session()
        (directory / "manifest.json.tmp").write_text("synthetic pending manifest", encoding="utf-8")

        def snapshot() -> dict:
            return {path.relative_to(self.root).as_posix():
                    (path.read_bytes() if path.is_file() else None, path.stat().st_mtime_ns)
                    for path in (self.root, *self.root.rglob("*"))}

        before = snapshot()
        original_open = Path.open

        def read_only_open(path: Path, mode: str = "r", *args, **kwargs):
            if mode not in ("r", "rb"):
                raise AssertionError("Reader attempted a writable file open")
            return original_open(path, mode, *args, **kwargs)

        with ExitStack() as stack:
            stack.enter_context(patch.object(Path, "open", read_only_open))
            for method in ("mkdir", "unlink", "rename", "replace", "touch", "write_text", "write_bytes"):
                stack.enter_context(patch.object(Path, method, side_effect=AssertionError("Reader attempted a filesystem mutation")))
            for method in ("open_session", "save_condition", "_write_manifest", "_atomic_write"):
                stack.enter_context(patch.object(FilesystemDatasetRepository, method,
                                                 side_effect=AssertionError("Reader invoked a capture mutation")))
            sessions = self.reader.discover_sessions()
            self.reader.read_manifest(directory)
            self.reader.read_condition(sessions[0], "center")
            samples = tuple(self.reader.iter_samples())
            self.assertEqual(len(samples), 4)
        self.assertTrue(snapshot() == before, "Source bytes, directory entries and modification times must remain unchanged")


if __name__ == "__main__":
    unittest.main()
