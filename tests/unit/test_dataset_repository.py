from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

from face_recognition.domain.models.dataset import CONDITIONS, COORDINATE_SPACE, CaptureProtocol
from face_recognition.infrastructure.persistence.dataset_repository import (
    FilesystemDatasetRepository, manifest_from_dict, manifest_to_dict, sample_from_json, sample_to_json,
)
from tests.dataset_fakes import condition_samples, session_config


class DatasetSerializationTests(unittest.TestCase):
    def test_jsonl_roundtrip_preserves_exact_values_and_landmark_order(self) -> None:
        config = session_config()
        for count in (1, 2, 17, 478, 500):
            with self.subTest(count=count):
                sample = condition_samples(config, landmark_count=count)[0]
                encoded = sample_to_json(sample)
                decoded = sample_from_json(encoded)
                self.assertEqual(decoded, sample)
                data = json.loads(encoded)
                self.assertEqual(data["landmarks"], [[p.x, p.y, p.z] for p in sample.landmarks.points])
                self.assertEqual(set(data), {"schema_version", "person_id", "session_id", "condition", "sample_index",
                                            "detector_timestamp_ms", "inference_width", "inference_height", "landmarks"})

    def test_nonfinite_json_and_malformed_landmark_shapes_are_rejected(self) -> None:
        data = json.loads(sample_to_json(condition_samples(session_config())[0]))
        for value in (float("nan"), float("inf"), float("-inf")):
            for axis in range(3):
                data["landmarks"] = [[0.1, 0.2, 0.3]]
                data["landmarks"][0][axis] = value
                with self.subTest(value=value, axis=axis), self.assertRaisesRegex(ValueError, "Non-finite"):
                    sample_from_json(json.dumps(data))
        for points in ([], [[0, 1]], [[0, 1, 2, 3]], [[True, 1, 2]], [["0", 1, 2]]):
            data["landmarks"] = points
            with self.subTest(points=points), self.assertRaises(ValueError):
                sample_from_json(json.dumps(data))

    def test_unknown_version_and_invalid_json_are_rejected(self) -> None:
        data = json.loads(sample_to_json(condition_samples(session_config())[0]))
        data["schema_version"] = 2
        with self.assertRaisesRegex(ValueError, "schema version"):
            sample_from_json(json.dumps(data))
        with self.assertRaises(ValueError):
            sample_from_json("not JSON")


class DatasetRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "raw"
        self.repository = FilesystemDatasetRepository(self.root)
        self.config = session_config()
        self.directory = self.root / "p001" / "session_01"
        self.manifest = self.repository.open_session(self.config)

    def test_manifest_creation_records_interpretation_metadata(self) -> None:
        data = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(data["schema_version"], 1)
        self.assertEqual(data["person_id"], "p001")
        self.assertEqual(data["camera_backend"], "opencv")
        self.assertEqual(data["camera_device_id"], "0")
        self.assertEqual((data["inference_width"], data["inference_height"]), (480, 270))
        self.assertEqual(data["landmark_coordinate_space"], COORDINATE_SPACE)
        self.assertEqual(data["conditions"], list(CONDITIONS))
        self.assertEqual(data["completed_conditions"], [])
        self.assertIsNone(data["expected_landmark_count"])
        self.assertIsNotNone(datetime.fromisoformat(data["created_at"]).utcoffset())
        self.assertEqual(manifest_from_dict(data), self.manifest)

    def test_completed_condition_roundtrip_and_manifest_tracking(self) -> None:
        samples = condition_samples(self.config, landmark_count=17)
        updated = self.repository.save_condition(self.manifest, "center", samples)
        self.assertEqual(updated.expected_landmark_count, 17)
        self.assertEqual(updated.completed_conditions, ("center",))
        self.assertEqual(self.repository.read_condition(updated, "center"), samples)
        self.assertFalse((self.directory / "center.jsonl.tmp").exists())
        self.assertFalse((self.directory / "manifest.json.tmp").exists())
        self.assertEqual(self.repository.open_session(self.config), updated)

    def test_partial_batch_is_not_published_or_marked_completed(self) -> None:
        with self.assertRaisesRegex(ValueError, "full configured sample count"):
            self.repository.save_condition(self.manifest, "center", condition_samples(self.config)[:19])
        self.assertFalse((self.directory / "center.jsonl").exists())
        self.assertEqual(self.repository.open_session(self.config).completed_conditions, ())

    def test_publication_failure_leaves_only_temporary_data_and_resume_restarts(self) -> None:
        with patch("face_recognition.infrastructure.persistence.dataset_repository.os.link", side_effect=OSError("publish failure")):
            with self.assertRaisesRegex(OSError, "publish failure"):
                self.repository.save_condition(self.manifest, "center", condition_samples(self.config))
        self.assertTrue((self.directory / "center.jsonl.tmp").exists())
        self.assertFalse((self.directory / "center.jsonl").exists())
        resumed = self.repository.open_session(self.config)
        self.assertEqual(resumed.completed_conditions, ())
        updated = self.repository.save_condition(resumed, "center", condition_samples(self.config))
        self.assertEqual(updated.completed_conditions, ("center",))

    def test_failed_atomic_manifest_update_keeps_old_manifest_and_recovers_full_file(self) -> None:
        before = (self.directory / "manifest.json").read_bytes()
        with patch("face_recognition.infrastructure.persistence.dataset_repository.os.replace", side_effect=OSError("manifest failure")):
            with self.assertRaisesRegex(OSError, "manifest failure"):
                self.repository.save_condition(self.manifest, "center", condition_samples(self.config))
        self.assertEqual((self.directory / "manifest.json").read_bytes(), before)
        self.assertTrue((self.directory / "center.jsonl").exists())
        resumed = self.repository.open_session(self.config)
        self.assertEqual(resumed.completed_conditions, ("center",))
        self.assertEqual(resumed.expected_landmark_count, 2)

    def test_existing_completed_condition_is_never_overwritten(self) -> None:
        updated = self.repository.save_condition(self.manifest, "center", condition_samples(self.config))
        before = (self.directory / "center.jsonl").read_bytes()
        with self.assertRaisesRegex(ValueError, "already completed"):
            self.repository.save_condition(updated, "center", condition_samples(self.config))
        self.assertEqual((self.directory / "center.jsonl").read_bytes(), before)

    def test_incompatible_metadata_and_protocol_are_rejected_without_changes(self) -> None:
        before = (self.directory / "manifest.json").read_bytes()
        for fields in ({"inference_width": 640}, {"orientation": "rotate180"}, {"camera_device_id": "1"},
                       {"camera_backend": "picamera2"}, {"detector_model_sha256": "b" * 64},
                       {"mediapipe_version": "different"}, {"protocol": CaptureProtocol(5, 21, 250)},
                       {"protocol": CaptureProtocol(6, 20, 250)}, {"protocol": CaptureProtocol(5, 20, 251)}):
            with self.subTest(fields=fields), self.assertRaisesRegex(ValueError, "Incompatible"):
                self.repository.open_session(replace(self.config, **fields))
        self.assertEqual((self.directory / "manifest.json").read_bytes(), before)

    def test_resume_checks_completed_file_integrity(self) -> None:
        self.repository.save_condition(self.manifest, "center", condition_samples(self.config))
        path = self.directory / "center.jsonl"
        path.write_text(path.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "full configured sample count"):
            self.repository.open_session(self.config)

    def test_partial_unmarked_final_file_is_rejected_not_adopted(self) -> None:
        (self.directory / "center.jsonl").write_text(sample_to_json(condition_samples(self.config)[0]) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "full configured sample count"):
            self.repository.open_session(self.config)

    def test_batch_metadata_count_spacing_and_order_validation(self) -> None:
        good = condition_samples(self.config)
        bad_batches = [(*good[:1], replace(good[1], sample_index=0), *good[2:]),
                       (*good[:1], replace(good[1], detector_timestamp_ms=good[0].detector_timestamp_ms), *good[2:]),
                       (*good[:1], replace(good[1], detector_timestamp_ms=good[0].detector_timestamp_ms + 249), *good[2:]),
                       (replace(good[0], person_id="p002"), *good[1:]),
                       (replace(good[0], inference_height=480), *good[1:]),
                       (condition_samples(self.config, landmark_count=3)[0], *good[1:])]
        for batch in bad_batches:
            with self.subTest(batch_type=type(batch)), self.assertRaises(ValueError):
                self.repository.save_condition(self.manifest, "center", batch)
        with self.assertRaisesRegex(ValueError, "protocol order"):
            self.repository.save_condition(self.manifest, "translation_left", condition_samples(self.config, "translation_left"))
        self.assertFalse((self.directory / "center.jsonl").exists())

    def test_manifest_version_and_condition_order_are_validated(self) -> None:
        data = manifest_to_dict(self.manifest)
        for fields in ({"schema_version": 2}, {"conditions": list(reversed(CONDITIONS))},
                       {"completed_conditions": ["translation_left"], "expected_landmark_count": 2},
                       {"landmark_coordinate_space": "identity-normalized"}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                manifest_from_dict({**data, **fields})

    def test_existing_session_without_manifest_is_not_silently_adopted(self) -> None:
        (self.root / "p001" / "another").mkdir()
        with self.assertRaisesRegex(ValueError, "no manifest"):
            self.repository.open_session(replace(self.config, session_id="another"))

    def test_symlinked_session_and_manifest_paths_are_rejected(self) -> None:
        for name in ("session_01", "manifest.json"):
            with self.subTest(name=name), patch.object(Path, "is_symlink", lambda path: path.name == name):
                with self.assertRaisesRegex(ValueError, "without symlinks"):
                    self.repository.open_session(self.config)

    def test_resolved_session_outside_root_is_rejected(self) -> None:
        with patch.object(Path, "resolve", return_value=self.root.parent / "outside"):
            with self.assertRaisesRegex(ValueError, "inside the dataset root"):
                self.repository.open_session(self.config)
