"""Dynamic captures and bundles verified only with synthetic technical identities."""

from dataclasses import asdict, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from face_recognition.application.recognition_config import metadata_sha256
from face_recognition.application.train_recognition_model import TrainRecognitionModel
from face_recognition.application.training_data import TrainingData
from face_recognition.domain.models.dataset import CONDITIONS
from face_recognition.domain.models.dynamic_population import DynamicDatasetProvenance, DynamicPopulation, DynamicTrainingProtocol, MODEL_ID
from face_recognition.domain.models.processed_sample import POSE_REPRESENTATION, SampleProvenance
from face_recognition.domain.models.rejection import RejectionError
from face_recognition.domain.models.training_dataset import CatalogFile, ObservedSessions, TrainingCatalog
from face_recognition.domain.models.v3_deployable_model import IdentityDecision
from face_recognition.infrastructure.classifiers.sklearn_recognition import RecognitionModelRunner
from face_recognition.infrastructure.features.pose_landmark_normalizer import POSE_NORMALIZATION
from face_recognition.infrastructure.persistence.artifact_integrity import artifact_fingerprints
from face_recognition.infrastructure.persistence.dynamic_configuration import dynamic_configuration, parse_population
from face_recognition.infrastructure.persistence.dynamic_training_dataset import (
    DynamicCaptureProvenance, DynamicTrainingInputs, _processed_hashes,
)
from face_recognition.infrastructure.persistence.recognition_model import (
    DYNAMIC_FILES, RuntimeCompatibilityError, load_recognition_bundle, publish_dynamic_bundle,
)
from face_recognition.infrastructure.persistence.training_dataset import read_training_catalog
from face_recognition.presentation.cli.train_dynamic_model import main
from tests.recognition_fakes import RecordingDeployableRunner


def population(known=2, unknown=1):
    return DynamicPopulation(tuple(f"p{i:03d}" for i in range(1, known + 1)),
                             tuple(f"p{i:03d}" for i in range(101, 101 + unknown)))


def samples(spec):
    return tuple(SampleProvenance(p, s, c, i, 1000 + 250*i, 480, 270)
                 for p, s, c in spec.condition_keys for i in range(20))


def memory_data(spec, *, real_fit=False):
    rows = samples(spec)
    labels = {p: i for i, p in enumerate(spec.known_ids)}
    y = np.array([labels.get(r.person_id, -1) for r in rows], dtype=np.int32)
    if real_fit:
        X = np.random.default_rng(80).normal(size=(len(y), 1434)).astype(np.float32)
        X[:, :80] += 4 * y[:, None]
        X[y == -1] += 20
    else:
        X = np.zeros((len(y), 1434), dtype=np.float32)
        X[:, 0] = y
    catalog = TrainingCatalog(POSE_REPRESENTATION, "a"*64, len(rows),
        tuple(ObservedSessions(p, s) for p, s in sorted(spec.assignments.items())),
        tuple(CatalogFile(p, s, c, f"{POSE_REPRESENTATION}/{p}/{s}/{c}.jsonl", 1)
              for p, s, c in spec.condition_keys))
    digest = metadata_sha256(catalog)
    config = dynamic_configuration(spec, DynamicDatasetProvenance("a"*64, "b"*64, digest, "c"*64, "d"*64, "e"*64))
    protocol = DynamicTrainingProtocol(spec, catalog, digest)
    return TrainingData(protocol, X, y, rows, "e"*64), config


class DynamicPopulationTests(unittest.TestCase):
    def test_ratio_table_and_unbounded_unknown_count(self):
        for k, u, valid in ((2,1,True), (10,4,False), (10,5,True), (10,8,True),
                            (11,5,False), (11,6,True), (11,8,True), (16,8,True),
                            (17,8,False), (17,9,True), (20,15,True), (11,50,True),
                            (0,0,False), (1,5,False), (2,0,False)):
            with self.subTest(known=k, unknown=u):
                if valid:
                    self.assertEqual(population(k,u).minimum_unknown, (k+1)//2)
                else:
                    with self.assertRaisesRegex(RejectionError, f"{u} UNKNOWN"):
                        population(k,u)

    def test_strict_schema_and_deterministic_roles(self):
        raw = json.loads(json.dumps(asdict(population())))
        raw["known_ids"].reverse()
        self.assertEqual(parse_population(raw), population())
        for change in ({"unknown_ids": ["p001"]}, {"known_ids": ["p001", "p001"]},
                       {"unknown_ids": ["P001"]}, {"unknown_ids": ["../bad"]},
                       {"schema_version": True}, {"extra": 1}, {"known_sessions": ["session_01"]},
                       {"representation": "all_xyz_v1"}, {"capture_protocol": {"samples_per_condition": 20}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                parse_population(dict(raw, **change))

    def test_eleven_known_eight_unknown_use_one_frozen_fit_and_all_calibration(self):
        spec = population(11,8)
        data, config = memory_data(spec)
        runner = RecordingDeployableRunner()
        result = TrainRecognitionModel(runner).train_dynamic(data, config)
        self.assertEqual(runner.fit_count, 1)
        self.assertEqual(result.model.class_order, spec.known_ids)
        self.assertEqual(runner.events[0][1:4], (spec.known_sessions[:3], spec.known_ids, 8580))
        self.assertEqual(runner.model_calls[0][2:], (spec.known_ids, ("session_04",)))
        self.assertEqual(runner.model_calls[1][2:], (spec.unknown_ids, spec.unknown_sessions))
        self.assertEqual({call[1] for call in runner.model_calls}, {id(result.model.fitted)})
        summary = result.summary
        self.assertEqual(summary.known_calibration.source_samples, 2860)
        self.assertEqual(summary.unknown_calibration.source_samples, 4160)
        self.assertEqual(summary.fingerprints_before_calibration, summary.fingerprints_after_calibration)


class DynamicDatasetTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.root = self.folder / "processed"
        self.spec = population()
        self.path = self.folder / "population.json"
        self.path.write_text(json.dumps(asdict(self.spec)), encoding="utf-8")

    def fixture(self, *, keys=None, short=False, corrupt=False, numeric=False):
        keys = self.spec.condition_keys if keys is None else keys
        n = 0
        for p, s, c in keys:
            path = self.root / POSE_REPRESENTATION / p / s / (c+".jsonl")
            path.parent.mkdir(parents=True, exist_ok=True)
            count = 19 if short and c == "center" else 20
            n += count
            with path.open("w", encoding="utf-8", newline="\n") as stream:
                if not numeric or corrupt and c == "center":
                    stream.write("invalid fixture\n")
                    continue
                for i in range(count):
                    row = asdict(SampleProvenance(p,s,c,i,1000+250*i,480,270))
                    row.update(schema_version=1, representation=POSE_REPRESENTATION, landmark_count=478,
                               feature_count=1434, source_relative_path=f"{p}/{s}/{c}.jsonl")
                    stream.write(json.dumps(row)[:-1] + ',"features":[' + "0,"*1433 + '0]}\n')
        common = dict(schema_version=1, completion_state="completed", participants=len({p for p,_,_ in keys}),
            sessions=len({(p,s) for p,s,_ in keys}), conditions=len(keys), total_samples_processed=n,
            rejected_samples=0, error_count=0, normalization="PoseLandmarkNormalizer",
            normalization_parameters=POSE_NORMALIZATION, feature_extraction="LandmarkFeatureExtractor",
            inference_dimensions="per_source_sample")
        manifest = self.root / "processing_manifest.json"
        manifest.write_text(json.dumps(dict(common, representations=[POSE_REPRESENTATION],
            condition_files_per_representation=len(keys), total_condition_files=len(keys))), encoding="utf-8")
        (self.root / POSE_REPRESENTATION / "processing_manifest.json").write_text(json.dumps(dict(common,
            representation=POSE_REPRESENTATION, condition_files=len(keys))), encoding="utf-8")
        catalog = read_training_catalog(self.root)
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        raw["dynamic_capture"] = dict(schema_version=1, population_sha256=metadata_sha256(self.spec),
                                      raw_dataset_sha256="a"*64,
                                      processed_files_sha256=metadata_sha256(_processed_hashes(self.root, catalog)))
        manifest.write_text(json.dumps(raw), encoding="utf-8")
        return self.root / catalog.condition_files[0].source_relative_path

    def test_preflight_reads_complete_valid_dataset_and_verifies_changed_bytes(self):
        first = self.fixture(numeric=True)
        args = ["--processed-root", str(self.root), "--population-config", str(self.path), "--validate-only"]
        self.assertEqual(main(args), 0)
        inputs = DynamicTrainingInputs(self.root, self.path)
        contents = first.read_bytes()
        first.write_bytes(contents.replace(b'"feature_count": 1434', b'"feature_count": 1433', 1))
        with self.assertRaisesRegex(RejectionError, "changed"):
            inputs.verify()
        with self.assertRaisesRegex(RejectionError, "changed"):
            DynamicTrainingInputs(self.root, self.path)

    def test_missing_known_session(self):
        self.fixture(keys=tuple(key for key in self.spec.condition_keys if key[:2] != ("p001","session_04")))
        with self.assertRaisesRegex(RejectionError, "Missing selected session"):
            DynamicTrainingInputs(self.root, self.path)

    def test_missing_unknown_session(self):
        self.fixture(keys=tuple(key for key in self.spec.condition_keys if key[:2] != ("p101","session_02")))
        with self.assertRaisesRegex(RejectionError, "Missing selected session"):
            DynamicTrainingInputs(self.root, self.path)

    def test_missing_condition(self):
        self.fixture(keys=self.spec.condition_keys[1:])
        with self.assertRaisesRegex(RejectionError, "capture conditions"):
            DynamicTrainingInputs(self.root, self.path)

    def test_incorrect_sample_total(self):
        self.fixture(short=True)
        with self.assertRaisesRegex(RejectionError, "sample total"):
            DynamicTrainingInputs(self.root, self.path)

    def test_corrupt_payload_even_with_matching_checksums(self):
        self.fixture()
        with self.assertRaisesRegex(RejectionError, "payload"):
            DynamicTrainingInputs(self.root, self.path).load()

    def test_unexpected_identity(self):
        self.fixture(keys=(*self.spec.condition_keys, ("p999","session_01","center")))
        with self.assertRaisesRegex(RejectionError, "Unexpected processed"):
            DynamicTrainingInputs(self.root, self.path)

    def test_fresh_processing_binds_raw_manifest_and_validated_conditions(self):
        from face_recognition.domain.models.dataset import CaptureProtocol
        from face_recognition.infrastructure.persistence.raw_dataset_reader import RawDatasetReader
        from face_recognition.infrastructure.persistence.dataset_repository import manifest_to_dict
        from tests.dataset_fakes import session_config
        from face_recognition.domain.models.dataset import SessionManifest

        raw_root = self.folder / "raw"
        for person, sessions in self.spec.assignments.items():
            for index, name in enumerate(sessions, 1):
                directory = raw_root / person / name
                directory.mkdir(parents=True)
                config = session_config(person_id=person, session_id=name, protocol=CaptureProtocol())
                manifest = SessionManifest(config, f"2026-01-{index:02d}T12:00:00+00:00", 478, CONDITIONS)
                (directory / "manifest.json").write_text(json.dumps(manifest_to_dict(manifest)), encoding="utf-8")
                for condition in CONDITIONS:
                    (directory / (condition+".jsonl")).write_bytes(b"synthetic bytes for digest only")
        provenance = DynamicCaptureProvenance(RawDatasetReader(raw_root), self.path)
        self.fixture()
        self.assertEqual(provenance(self.root)["raw_dataset_sha256"], provenance.raw_digest)
        (raw_root / "p001/session_01/center.jsonl").write_bytes(b"changed")
        with self.assertRaisesRegex(RejectionError, "changed"):
            provenance(self.root)


class DynamicBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temp = TemporaryDirectory()
        cls.addClassCleanup(temp.cleanup)
        cls.root = Path(temp.name)
        cls.data, cls.config = memory_data(population(), real_fit=True)
        runner = RecognitionModelRunner(cls.config)
        cls.result = TrainRecognitionModel(runner).train_dynamic(cls.data, cls.config)

    def publish(self, name):
        target = self.root / name
        manifest = publish_dynamic_bundle(self.result, target, (), lambda: None)
        return target, manifest

    def test_roundtrip_atomic_no_overwrite_and_unknown_suppression(self):
        target, manifest = self.publish("roundtrip")
        self.assertEqual(manifest["model_id"], MODEL_ID)
        self.assertEqual(set(p.name for p in target.iterdir()), set(DYNAMIC_FILES))
        self.assertEqual(manifest["dynamic_configuration"]["unknown_count"], 1)
        before = artifact_fingerprints(target, DYNAMIC_FILES)
        with self.assertRaises(ValueError):
            publish_dynamic_bundle(self.result, target, (), lambda: None)
        self.assertEqual(before, artifact_fingerprints(target, DYNAMIC_FILES))
        recognizer = load_recognition_bundle(target)
        self.assertEqual(recognizer.predict_window(self.data.X[:7]).identity, "p001")
        unknown = recognizer.predict_window(self.data.X[-7:])
        self.assertEqual(unknown.decision, IdentityDecision.UNKNOWN)
        self.assertIsNone(unknown.identity)

    def test_incomplete_tampered_and_runtime_mismatch_are_rejected(self):
        target, _ = self.publish("damaged")
        manifest_path = target / "manifest.json"
        original = manifest_path.read_bytes()
        manifest = json.loads(original)
        manifest["calibration"]["versions"]["python"] = "99.0.0"
        manifest["creation"]["versions"]["python"] = "99.0.0"
        manifest["manifest_sha256"] = ""
        manifest["manifest_sha256"] = metadata_sha256(manifest)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(RuntimeCompatibilityError):
            load_recognition_bundle(target)
        manifest_path.write_bytes(original)
        model = target / "model.joblib"
        with model.open("ab") as stream:
            stream.write(b"tampered")
        with self.assertRaises(RejectionError):
            load_recognition_bundle(target)
        (target/"manifest.json").unlink()
        with self.assertRaises(RejectionError):
            load_recognition_bundle(target)

    def test_late_source_failure_leaves_no_completed_or_staged_directory(self):
        target = self.root / "late"
        def fail():
            self.assertFalse(target.exists())
            raise RejectionError("changed source")
        with self.assertRaises(RejectionError):
            publish_dynamic_bundle(self.result, target, (), fail)
        self.assertFalse(target.exists())
        self.assertFalse(list(self.root.glob("late.*.tmp")))

    def test_atomic_rename_does_not_replace_target_created_after_preflight(self):
        target = self.root / "concurrent"
        with self.assertRaises(RejectionError):
            publish_dynamic_bundle(self.result, target, (), target.mkdir)
        self.assertTrue(target.is_dir())
        self.assertEqual(list(target.iterdir()), [])
        self.assertFalse(list(self.root.glob("concurrent.*.tmp")))
