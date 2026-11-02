from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path, PureWindowsPath
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import warnings

import numpy as np

from face_recognition.application.v3_deployable_recognition import V3DeployableRecognizer
from face_recognition.application.create_v3_cv_protocol import metadata_sha256
from face_recognition.application.train_v3_deployable_model import TrainV3DeployableModel
from face_recognition.domain.models.rejection import RejectionError
from face_recognition.infrastructure.classifiers.sklearn_v3_deployable import SklearnV3DeployableRunner, fitted_v3_fingerprints, validate_fitted_v3_state
from face_recognition.infrastructure.persistence.artifact_integrity import artifact_fingerprints
from face_recognition.infrastructure.persistence.v3_deployable_model import (
    CONFIG_FILE, FILES, MANIFEST_FILE, MODEL_FILE, load_v3_deployable_bundle, publish_v3_deployable_bundle,
)
from face_recognition.infrastructure.persistence.v3_open_set_config import read_frozen_v3_configuration
from tests.v3_deployable_fakes import deployable_fixture


class V3DeployableRepositoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory(prefix="synthetic_v3_bundle_")
        cls.folder = Path(cls.temporary.name)
        cls.data, cls.config, cls.provenance, _, cls.config_path, *_ = deployable_fixture(cls.folder)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning); warnings.simplefilter("ignore", RuntimeWarning)
            cls.result = TrainV3DeployableModel(SklearnV3DeployableRunner(cls.config)).train(cls.data, cls.config, cls.provenance)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def publish(self, name):
        path = self.folder / name
        manifest = publish_v3_deployable_bundle(self.result, self.config_path, path)
        return path, manifest

    def test_full_roundtrip_dimensions_classes_threshold_and_state_unchanged(self):
        before = fitted_v3_fingerprints(self.result.model.fitted)
        path, manifest = self.publish("roundtrip")
        self.assertEqual({p.name for p in path.iterdir()}, set(FILES))
        self.assertEqual(manifest["input_features"], 1434)
        self.assertEqual(manifest["selected_features"], 448)
        self.assertEqual(manifest["representation"], "all_xyz_pose_v2")
        self.assertEqual(manifest["threshold"], self.result.model.threshold)
        self.assertEqual(manifest["class_order"], self.result.model.class_order)
        service = load_v3_deployable_bundle(path)
        result = service.predict_window(self.data.X[:7])
        self.assertEqual(result.threshold, self.result.model.threshold)
        self.assertEqual(fitted_v3_fingerprints(self.result.model.fitted), before)

    def test_loader_never_fits_calibrates_or_performs_inference(self):
        path, _ = self.publish("load_only")
        forbidden = (
            "face_recognition.infrastructure.classifiers.sklearn_v3_deployable.SklearnV3DeployableRunner.fit",
            "face_recognition.infrastructure.classifiers.sklearn_v3_deployable.SklearnV3DeployableRunner.infer",
            "face_recognition.application.rejection_metrics.calibrate_threshold",
        )
        with ExitStack() as stack:
            for name in forbidden: stack.enter_context(patch(name, side_effect=AssertionError("No work on load")))
            self.assertIsNotNone(load_v3_deployable_bundle(path))

    def test_loading_still_validates_structure_and_full_fitted_fingerprints(self):
        path, _ = self.publish("startup_full_validation")
        with patch("face_recognition.infrastructure.persistence.v3_deployable_model.validate_fitted_v3_state", wraps=validate_fitted_v3_state) as structure, \
                patch("face_recognition.infrastructure.persistence.v3_deployable_model.fitted_v3_fingerprints", wraps=fitted_v3_fingerprints) as fingerprints:
            load_v3_deployable_bundle(path)
        structure.assert_called_once()
        fingerprints.assert_called_once()

    def test_manifest_or_frozen_config_tampering_rejected_before_deserialization(self):
        for filename in (MANIFEST_FILE, CONFIG_FILE):
            path, _ = self.publish("tamper_" + filename)
            artifact = path / filename
            raw = json.loads(artifact.read_text(encoding="utf-8"))
            if filename == MANIFEST_FILE:
                raw["threshold"] += 1.
            else:
                raw["pipeline"]["selected_features"] = 447
            artifact.write_text(json.dumps(raw), encoding="utf-8")
            with self.subTest(filename=filename), patch("face_recognition.infrastructure.persistence.v3_deployable_model.joblib.load") as loader:
                with self.assertRaises(RejectionError): load_v3_deployable_bundle(path)
                loader.assert_not_called()

    def test_incompatible_frozen_policy_rejected_even_with_updated_file_checksums(self):
        path, manifest = self.publish("incompatible_frozen_policy")
        config_path = path / CONFIG_FILE
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["pipeline"]["selected_features"] = 447
        config["configuration_sha256"] = ""
        config["configuration_sha256"] = metadata_sha256(config)
        config_path.write_text(json.dumps(config), encoding="utf-8")
        hashes = artifact_fingerprints(path, (MODEL_FILE, CONFIG_FILE))
        manifest["artifacts"] = [dict(filename=name, sha256=digest) for name, digest in hashes]
        manifest["frozen_configuration"]["source_file_sha256"] = dict(hashes)[CONFIG_FILE]
        manifest["manifest_sha256"] = ""
        manifest["manifest_sha256"] = metadata_sha256(manifest)
        (path / MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")
        with patch("face_recognition.infrastructure.persistence.v3_deployable_model.read_frozen_v3_configuration", wraps=read_frozen_v3_configuration) as validate, \
                patch("face_recognition.infrastructure.persistence.v3_deployable_model.joblib.load") as loader:
            with self.assertRaises(RejectionError): load_v3_deployable_bundle(path)
            validate.assert_called_once()
            loader.assert_not_called()

    def test_loaded_inference_matches_old_deep_checked_path_without_runtime_hashing(self):
        class DeepCheckedRecognizer(V3DeployableRecognizer):
            def _verify(self):
                super()._verify()
                if self._runner.fingerprints(self._model.fitted) != self._expected:
                    raise RejectionError("Reference fitted state changed")

        path, _ = self.publish("hot_path_equivalence")
        service = load_v3_deployable_bundle(path)
        reference = DeepCheckedRecognizer(service._model, SklearnV3DeployableRunner(self.config),
            self.result.summary.fingerprints_after_calibration)
        expected = tuple(reference.infer_observation(row) for row in self.data.X[:7])
        expected_decision = reference.decide_observations(expected)
        forbidden = (
            "face_recognition.infrastructure.classifiers.sklearn_v3_deployable.SklearnV3DeployableRunner.fingerprints",
            "face_recognition.infrastructure.classifiers.sklearn_v3_deployable.fitted_v3_fingerprints",
            "face_recognition.infrastructure.classifiers.v3_fitted_integrity.fitted_value_sha256",
            "face_recognition.infrastructure.persistence.v3_deployable_model.artifact_fingerprints",
            "pickle.dump",
        )
        with ExitStack() as stack:
            for name in forbidden: stack.enter_context(patch(name, side_effect=AssertionError("No hot-path model hashing/serialization")))
            actual = tuple(service.infer_observation(row) for row in self.data.X[:7])
            outcome = service.decide_observations(actual)
        for before, after in zip(expected, actual):
            self.assertEqual(before.candidate_index, after.candidate_index)
            self.assertEqual(before.candidate_identity, after.candidate_identity)
            np.testing.assert_allclose(before.decision_scores, after.decision_scores, rtol=0, atol=0)
            self.assertEqual(before.knownness_score, after.knownness_score)
        self.assertEqual(outcome, expected_decision)

    def test_artifact_tampering_or_missing_manifest_rejected_before_deserialization(self):
        path, _ = self.publish("tamper")
        artifact = path / MODEL_FILE
        artifact.write_bytes(artifact.read_bytes() + b"tampered")
        with patch("face_recognition.infrastructure.persistence.v3_deployable_model.joblib.load") as loader:
            with self.assertRaises(RejectionError): load_v3_deployable_bundle(path)
            loader.assert_not_called()
        (path / MANIFEST_FILE).unlink()
        with self.assertRaises(RejectionError): load_v3_deployable_bundle(path)

    def test_completed_directory_overwrite_refused_and_files_unchanged(self):
        path, _ = self.publish("immutable")
        before = artifact_fingerprints(path, FILES)
        with self.assertRaises(ValueError): publish_v3_deployable_bundle(self.result, self.config_path, path)
        self.assertEqual(artifact_fingerprints(path, FILES), before)

    def test_serialization_failure_or_late_input_change_removes_staging_and_final_output(self):
        for defect in ("serialization", "inputs"):
            output = self.folder / ("failed_" + defect)
            with ExitStack() as stack:
                if defect == "serialization":
                    stack.enter_context(patch("face_recognition.infrastructure.persistence.v3_deployable_model.joblib.dump", side_effect=OSError("synthetic")))
                callback = (lambda: (_ for _ in ()).throw(RejectionError("changed"))) if defect == "inputs" else (lambda: None)
                with self.assertRaises(RejectionError):
                    publish_v3_deployable_bundle(self.result, self.config_path, output, verify_inputs=callback)
            self.assertFalse(output.exists())
            self.assertEqual(tuple(self.folder.glob(output.name + ".*.tmp")), ())

    def test_changed_model_threshold_fingerprint_or_generation_cannot_be_published(self):
        for summary in (replace(self.result.summary, threshold=float("nan")),
                        replace(self.result.summary, fit_generation_after_calibration=2),
                        replace(self.result.summary, fingerprints_after_calibration=replace(self.result.summary.fingerprints_after_calibration, overall_sha256="0" * 64))):
            with self.assertRaises(RejectionError):
                publish_v3_deployable_bundle(replace(self.result, summary=summary), self.config_path, self.folder / "bad_state")
        self.assertFalse((self.folder / "bad_state").exists())

    def test_manifest_is_deterministic_aggregate_only_and_calibration_labeled(self):
        first, manifest = self.publish("deterministic_a")
        second, _ = self.publish("deterministic_b")
        self.assertEqual((first / MANIFEST_FILE).read_bytes(), (second / MANIFEST_FILE).read_bytes())
        raw = json.loads((first / MANIFEST_FILE).read_text(encoding="utf-8"))
        self.assertIn("CALIBRATION METRICS", raw["calibration"]["metrics_scope"])
        self.assertIn("V2 remains", raw["calibration"]["limitation"])
        self.assertEqual(len(raw["artifacts"]), 2)
        def check(value):
            if isinstance(value, dict):
                self.assertFalse(set(value) & {"features", "landmarks", "support_vectors_", "selected_feature_indices", "_centers", "_factors", "individual_scores", "predicted"})
                for child in value.values(): check(child)
            elif isinstance(value, list):
                for child in value: check(child)
            elif isinstance(value, str): self.assertFalse(Path(value).is_absolute() or PureWindowsPath(value).is_absolute())
        check(raw)
