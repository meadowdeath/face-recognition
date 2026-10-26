from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from hashlib import sha256
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import joblib
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from face_recognition.application.refit_v2_open_set_model import V2OpenSetRefitter
from face_recognition.domain.models.open_set_model import OpenSetModelError
from face_recognition.infrastructure.classifiers.sklearn_v2_open_set_refit import SklearnV2OpenSetRefitRunner
from face_recognition.infrastructure.persistence.model_repository import _hash
from face_recognition.infrastructure.persistence.v2_model_repository import V2_ARTIFACTS, load_v2_open_set_model, persist_v2_open_set_model
from face_recognition.infrastructure.persistence.v2_study_loader import V2StudyLoader
from face_recognition.presentation.cli.refit_v2_open_set_model import main
from tests.v2_open_set_refit_fakes import v2_refit_fixture, load_v2_refit_fixture


class V2ModelRepositoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory(prefix="synthetic_v2_repository_")
        cls.directory = Path(cls.temporary.name)
        cls.fixture = v2_refit_fixture(cls.directory)
        cls.data = load_v2_refit_fixture(cls.fixture)
        cls.config, cls.digest = cls.fixture[-2:]
        cls.result = V2OpenSetRefitter(SklearnV2OpenSetRefitRunner()).run(cls.config, cls.digest, cls.data)
        cls.inputs = (cls.fixture[0], cls.fixture[3], cls.fixture[4], cls.fixture[5])

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def publish(self):
        path = self.directory / self._testMethodName
        persist_v2_open_set_model(self.result, self.config, self.digest, path, self.inputs)
        return path

    def test_round_trip_hashes_exact_artifact_set_and_no_reference(self):
        path = self.publish()
        model = load_v2_open_set_model(path, self.config, self.digest)
        self.assertEqual({p.name for p in path.iterdir()}, {"manifest.json", *V2_ARTIFACTS})
        self.assertFalse((path / "novelty_reference.npz").exists())
        self.assertEqual(model.threshold, self.result.summary.analysis.calibration.threshold)
        self.assertEqual(model.parts.class_ids, self.result.parts.class_ids)
        self.assertEqual(model.parts.svm.shape_fit_, (144, 448))
        self.assertEqual(model.manifest["completion_state"], "completed")
        for name, metadata in model.manifest["artifacts"].items():
            self.assertEqual(metadata["sha256"], _hash(path / name))

    def test_loader_calls_no_inference_or_fitting(self):
        path = self.publish()
        with patch.object(SelectKBest, "fit", side_effect=AssertionError("no fit")), \
                patch.object(SelectKBest, "transform", side_effect=AssertionError("no transform")), \
                patch.object(StandardScaler, "transform", side_effect=AssertionError("no transform")), \
                patch.object(SVC, "predict", side_effect=AssertionError("no predict")), \
                patch.object(SVC, "decision_function", side_effect=AssertionError("no decisions")):
            self.assertEqual(load_v2_open_set_model(path, self.config, self.digest).threshold,
                             self.result.summary.analysis.calibration.threshold)

    def test_tampering_rejected_before_deserialization(self):
        path = self.publish()
        artifact = path / "identity_svm.joblib"
        artifact.write_bytes(artifact.read_bytes() + b"tampered")
        with patch("face_recognition.infrastructure.persistence.v2_model_repository.joblib.load") as read:
            with self.assertRaises(OpenSetModelError):
                load_v2_open_set_model(path, self.config, self.digest)
            read.assert_not_called()

    def test_missing_artifact_rejected(self):
        path = self.publish()
        (path / "threshold.json").unlink()
        with self.assertRaises(OpenSetModelError):
            load_v2_open_set_model(path, self.config, self.digest)

    def test_wrong_frozen_provenance_rejected(self):
        path = self.publish()
        with self.assertRaises(OpenSetModelError):
            load_v2_open_set_model(path, self.config, "c" * 64)

    def test_incompatible_metadata_and_threshold_rejected(self):
        path = self.publish()
        manifest_path = path / "manifest.json"
        original = json.loads(manifest_path.read_text())
        for key, value in (("status", "incomplete"), ("rejection_score", "mean_5nn"), ("window_size", 5),
                            ("identity_consistency_gate", True), ("threshold", float("nan")), ("selected_feature_count", 512)):
            bad = deepcopy(original)
            bad[key] = value
            manifest_path.write_text(json.dumps(bad))
            with self.assertRaises(OpenSetModelError):
                load_v2_open_set_model(path, self.config, self.digest)
        manifest_path.write_text(json.dumps(original))
        threshold = json.loads((path / "threshold.json").read_text())
        threshold["threshold"] += .01
        (path / "threshold.json").write_text(json.dumps(threshold))
        original["artifacts"]["threshold.json"]["sha256"] = _hash(path / "threshold.json")
        manifest_path.write_text(json.dumps(original))
        with self.assertRaises(OpenSetModelError):
            load_v2_open_set_model(path, self.config, self.digest)

    def test_wrong_fitted_dimension_even_with_updated_hash_rejected(self):
        path = self.publish()
        scaler = joblib.load(path / "scaler.joblib")
        scaler.n_features_in_ = 10
        joblib.dump(scaler, path / "scaler.joblib")
        metadata = json.loads((path / "manifest.json").read_text())
        metadata["artifacts"]["scaler.joblib"]["sha256"] = _hash(path / "scaler.joblib")
        (path / "manifest.json").write_text(json.dumps(metadata))
        with self.assertRaises(OpenSetModelError):
            load_v2_open_set_model(path, self.config, self.digest)

    def test_overwrite_refused_without_mutation(self):
        path = self.publish()
        before = {p.name: _hash(p) for p in path.iterdir()}
        with self.assertRaises(OpenSetModelError):
            persist_v2_open_set_model(self.result, self.config, self.digest, path, self.inputs)
        self.assertEqual(before, {p.name: _hash(p) for p in path.iterdir()})

    def test_manifest_written_last_and_failure_not_completed(self):
        from face_recognition.infrastructure.persistence.v2_model_repository import _write_json
        names = []
        def write(path, value):
            names.append(path.name)
            return _write_json(path, value)
        path = self.directory / "last_marker"
        with patch("face_recognition.infrastructure.persistence.v2_model_repository._write_json", write):
            persist_v2_open_set_model(self.result, self.config, self.digest, path, self.inputs)
        self.assertEqual(names, ["threshold.json", "manifest.json"])
        target = self.directory / "failed_stage"
        with patch("face_recognition.infrastructure.persistence.v2_model_repository.joblib.dump", side_effect=OSError("synthetic failure")):
            with self.assertRaises(OpenSetModelError):
                persist_v2_open_set_model(self.result, self.config, self.digest, target, self.inputs)
        self.assertFalse(target.exists())
        stages = list(self.directory.glob("failed_stage.*.tmp"))
        self.assertEqual(len(stages), 1)
        self.assertFalse((stages[0] / "manifest.json").exists())
        self.assertTrue((stages[0] / "INCOMPLETE.json").exists())

    def test_metadata_privacy_and_temporal_threshold_policy(self):
        path = self.publish()
        for name in ("manifest.json", "threshold.json"):
            text = (path / name).read_text()
            for forbidden in ('"features"', '"landmarks"', '"support_vectors"', '"per_sample_scores"',
                              '"frame_scores"', '"window_scores"', str(self.directory), "C:\\\\Users"):
                self.assertFalse(forbidden in text, f"Unexpected metadata content: {forbidden}")
        threshold = json.loads((path / "threshold.json").read_text())
        self.assertEqual((threshold["rejection_score"], threshold["threshold_type"], threshold["score_orientation"],
                          threshold["temporal_aggregation"], threshold["window_size"], threshold["calibration_method"]),
                         ("svm_top1", "global", "higher_is_known", "median", 7, "cross_session_oof"))
        self.assertEqual(threshold["selection_rule"], list(self.config.final_threshold_calibration_policy.selection_rule))


class V2RefitCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="synthetic_v2_refit_cli_")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.fixture = v2_refit_fixture(self.directory)
        self.output = self.directory / "model"

    def args(self):
        p, _, _, identity, unknown, config, _, _ = self.fixture
        return ["--processed-root", str(p), "--identity-split", str(identity), "--unknown-split", str(unknown),
                "--frozen-config", str(config), "--output-dir", str(self.output)]

    def test_cli_success_aggregate_sources_unchanged_and_overwrite_refused(self):
        before = {p: (sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns) for p in self.directory.rglob("*") if p.is_file()}
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            self.assertEqual(main(self.args()), 0)
        self.assertEqual(errors.getvalue(), "")
        self.assertIn("V2 FINAL EVALUATION HAS NOT BEEN PERFORMED", output.getvalue())
        self.assertIn("UNKNOWN unique windows=10; UNKNOWN fold observations=30", output.getvalue())
        self.assertIn("Direct-transfer validation=passed", output.getvalue())
        self.assertNotIn("known_a", output.getvalue())
        for path, fingerprint in before.items():
            self.assertEqual((sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns), fingerprint)
        with patch.object(V2StudyLoader, "load_frozen_development") as load, redirect_stderr(io.StringIO()):
            self.assertEqual(main(self.args()), 1)
            load.assert_not_called()

    def test_future_session_or_identity_rejected_before_any_payload_open(self):
        p, source, partition, _, _, _, config, _ = self.fixture
        loader = V2StudyLoader(p)
        for person, session in (("known_a", "future_session"), ("new_unknown", "visit")):
            path = loader.root / person / session
            path.mkdir(parents=True)
            (path / "center.jsonl").write_text("DO NOT READ THIS PAYLOAD")
            with patch.object(loader, "_read_references", side_effect=AssertionError("No feature opens")) as read:
                with self.assertRaises(ValueError):
                    loader.load_frozen_development(source, partition, config)
                read.assert_not_called()
            (path / "center.jsonl").unlink()
            path.rmdir()
            if person == "new_unknown":
                path.parent.rmdir()

    def test_bad_frozen_fields_rejected_before_data_load(self):
        path = self.fixture[5]
        original = json.loads(path.read_text())
        for field, value in (("rejection_score_family", "mean_5nn"), ("temporal_window_size", 5), ("identity_consistency_gate", True)):
            bad = deepcopy(original)
            bad["architecture"][field] = value
            path.write_text(json.dumps(bad))
            with patch.object(V2StudyLoader, "load_frozen_development") as load, redirect_stderr(io.StringIO()):
                self.assertEqual(main(self.args()), 1)
                load.assert_not_called()
        path.write_text(json.dumps({"configuration_id": "open_set_config_v1"}))
        with redirect_stderr(io.StringIO()):
            self.assertEqual(main(self.args()), 1)

    def test_unsafe_transfer_no_publication(self):
        errors = io.StringIO()
        with patch("face_recognition.application.refit_v2_open_set_model.validate_direct_transfer",
                   side_effect=OpenSetModelError("Frozen V2 transfer assumption failed; new development protocol required")), redirect_stderr(errors):
            self.assertEqual(main(self.args()), 1)
        self.assertFalse(self.output.exists())
        self.assertIn("assumption failed", errors.getvalue())

    def test_parameter_overrides_not_available(self):
        for extra in (("--force",), ("--threshold", ".5"), ("--k", "512")):
            with redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as result:
                    main(self.args() + list(extra))
            self.assertEqual(result.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
