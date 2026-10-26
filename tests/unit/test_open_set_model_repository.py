from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from face_recognition.application.refit_open_set_model import OpenSetModelRefitter
from face_recognition.config.settings import PROJECT_ROOT
from face_recognition.domain.models.open_set_model import OpenSetModelError
from face_recognition.infrastructure.classifiers.sklearn_open_set_refit import SklearnOpenSetRefitRunner
from face_recognition.infrastructure.persistence import model_repository as repository
from tests.open_set_refit_fakes import refit_fixture, load_refit_fixture


class OpenSetModelRepositoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory(prefix="synthetic_fitted_artifacts_")
        cls.directory = Path(cls.temporary.name)
        cls.fixture = refit_fixture(cls.directory)
        known, unknown, cls.frozen = load_refit_fixture(cls.fixture)
        cls.result = OpenSetModelRefitter(SklearnOpenSetRefitRunner()).run(cls.frozen.configuration, cls.frozen.sha256, known, unknown)
        cls.inputs = (cls.fixture[0], cls.fixture[3], cls.fixture[4], cls.fixture[5])

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def publish(self, name):
        path = self.directory / name
        repository.persist_open_set_model(self.result, self.frozen, path, self.inputs)
        return path

    def test_round_trip_hashes_metadata_and_no_prediction_when_loading(self):
        path = self.publish("round_trip")
        self.assertEqual({p.name for p in path.iterdir()}, {*repository.ARTIFACTS, "manifest.json"})
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        for name in repository.ARTIFACTS:
            self.assertEqual(manifest["artifacts"][name]["sha256"], sha256((path / name).read_bytes()).hexdigest())
        with patch("sklearn.svm.SVC.predict", side_effect=AssertionError("Loading cannot predict")), \
             patch("sklearn.preprocessing.StandardScaler.transform", side_effect=AssertionError("Loading cannot transform")):
            loaded = repository.load_open_set_model(path, self.frozen)
        self.assertEqual(loaded.threshold, self.result.summary.analysis.calibration.threshold)
        self.assertEqual(manifest["status"], "fitted_and_calibrated")
        self.assertEqual(manifest["completion_state"], "completed")
        self.assertEqual(loaded.parts.class_ids, self.result.parts.class_ids)
        original, labels = self.result.parts.reference.reference_arrays()
        actual, actual_labels = loaded.parts.reference.reference_arrays()
        self.assertTrue(np.array_equal(original, actual) and np.array_equal(labels, actual_labels))
        self.assertTrue(np.array_equal(loaded.parts.svm.support_vectors_, self.result.parts.svm.support_vectors_))
        self.assertNotIn("array(", repr(loaded))

    def test_manifest_last_and_validation_happens_before_completion_marker(self):
        writes, validated = [], []
        original_write, original_load = repository._write_json, repository._load_artifacts

        def write(path, data):
            writes.append(path.name)
            return original_write(path, data)

        def load(root, manifest, frozen):
            self.assertFalse((root / "manifest.json").exists())
            validated.append(True)
            return original_load(root, manifest, frozen)

        with patch.object(repository, "_write_json", write), patch.object(repository, "_load_artifacts", load):
            self.publish("manifest_last")
        self.assertEqual(writes, ["threshold.json", "manifest.json"])
        self.assertEqual(validated, [True])

    def test_missing_or_tampered_artifacts_rejected_before_unpickling(self):
        for kind in ("missing", "tampered"):
            path = self.publish(kind)
            artifact = path / "scaler.joblib"
            if kind == "missing":
                artifact.unlink()
            else:
                artifact.write_bytes(b"SYNTHETIC_TAMPER")
            with patch.object(repository.joblib, "load") as load:
                with self.assertRaises(OpenSetModelError):
                    repository.load_open_set_model(path, self.frozen)
                load.assert_not_called()

    def test_wrong_dimensions_rejected_even_with_updated_checksum(self):
        path = self.publish("wrong_dimensions")
        file = path / "novelty_reference.npz"
        with np.load(file, allow_pickle=False) as data:
            reference, labels, ids = data["reference"], data["reference_labels"], data["class_ids"]
        np.savez_compressed(file, reference=reference[:, :-1], reference_labels=labels, class_ids=ids)
        manifest_path = path / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["artifacts"][file.name]["sha256"] = sha256(file.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(OpenSetModelError):
            repository.load_open_set_model(path, self.frozen)

    def test_config_provenance_window_or_nonfinite_threshold_rejected(self):
        path = self.publish("provenance")
        with self.assertRaises(OpenSetModelError):
            repository.load_open_set_model(path, replace(self.frozen, sha256="f" * 64))
        original = (path / "manifest.json").read_text(encoding="utf-8")
        for field, value in (("threshold", float("nan")), ("selected_feature_count", 447), ("status", "incomplete")):
            manifest = json.loads(original)
            manifest[field] = value
            (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(OpenSetModelError):
                repository.load_open_set_model(path, self.frozen)

    def test_no_vectors_names_or_paths_in_manifest_reference_contains_only_needed_arrays(self):
        path = self.publish("privacy")
        text = (path / "manifest.json").read_text(encoding="utf-8")
        for marker in (str(self.directory), '"known_a"', '"known_b"', '"unknown_cal"', '"unknown_hold_a"'):
            self.assertNotIn(marker, text)
        manifest = json.loads(text)

        def inspect(value):
            if isinstance(value, dict):
                self.assertFalse(set(value) & {"features", "landmarks", "support_vectors", "support_vectors_", "reference_matrix",
                                               "individual_distances", "person_id", "session_id", "sample_index", "label_order"})
                for child in value.values():
                    inspect(child)
            elif isinstance(value, list):
                for child in value:
                    inspect(child)
            elif isinstance(value, str):
                self.assertFalse(Path(value).is_absolute())
        inspect(manifest)
        with np.load(path / "novelty_reference.npz", allow_pickle=False) as data:
            self.assertEqual(set(data.files), {"reference", "reference_labels", "class_ids"})

    def test_completed_output_refused_and_unsafe_location_rejected(self):
        path = self.publish("no_overwrite")
        before = (path / "manifest.json").read_bytes()
        with self.assertRaisesRegex(OpenSetModelError, "already exists"):
            repository.persist_open_set_model(self.result, self.frozen, path, self.inputs)
        self.assertTrue(before == (path / "manifest.json").read_bytes())
        for target in (PROJECT_ROOT / "models/invalid", self.fixture[0] / "invalid"):
            with self.assertRaises(OpenSetModelError):
                repository.validate_model_target(target, self.inputs)

    def test_failed_or_interrupted_staging_never_appears_complete(self):
        for kind, failure in (("failed_stage", OSError("private")), ("interrupted_stage", KeyboardInterrupt())):
            output = self.directory / kind
            with patch.object(repository.os, "rename", side_effect=failure):
                with self.assertRaises(OpenSetModelError if kind == "failed_stage" else KeyboardInterrupt):
                    repository.persist_open_set_model(self.result, self.frozen, output, self.inputs)
            self.assertFalse(output.exists())
            stages = list(self.directory.glob(kind + ".*.tmp"))
            self.assertEqual(len(stages), 1)
            self.assertFalse((stages[0] / "manifest.json").exists())
            self.assertEqual(json.loads((stages[0] / "INCOMPLETE.json").read_text(encoding="utf-8"))["usable"], False)
            with self.assertRaises(OpenSetModelError):
                repository.load_open_set_model(stages[0], self.frozen)
