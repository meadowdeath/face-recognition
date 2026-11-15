"""One synthetic operational fixture; no experiment execution or real data."""

from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
from platform import python_version
from types import SimpleNamespace

import numpy as np
import scipy
import sklearn

from face_recognition.application.recognition_config import metadata_sha256
from face_recognition.application.training_data import FrameScores, TrainingData
from face_recognition.domain.models.classification import EvaluationVersions
from face_recognition.domain.models.dataset import CONDITIONS
from face_recognition.domain.models.processed_sample import POSE_REPRESENTATION, SampleProvenance
from face_recognition.domain.models.training_dataset import TrainingSpecification, TrainingCatalog, CatalogFile, ObservedSessions
from face_recognition.domain.models.v3_deployable_model import FittedV3Fingerprints, V3TrainingProvenance
from face_recognition.domain.models.v3_open_set_config import (
    EVIDENCE_KINDS, EVIDENCE_SOURCE_TYPES, DevelopmentMetricEvidence, FrozenEvidenceReference, FrozenV3OpenSetConfiguration,
)
from face_recognition.infrastructure.classifiers.model_integrity import fitted_value_sha256
from face_recognition.infrastructure.features.pose_landmark_normalizer import POSE_NORMALIZATION
from face_recognition.infrastructure.persistence.artifact_integrity import file_sha256
from face_recognition.infrastructure.persistence.recognition_config import (
    current_implementation_fingerprints, current_pose_normalization, frozen_configuration_to_json,
)
from face_recognition.infrastructure.persistence.training_dataset import read_training_catalog, read_training_protocol, load_training_payloads


def deployable_fixture(folder, *, numeric=False):
    folder, samples = Path(folder), 8
    spec = TrainingSpecification(("known_00", "known_01"), tuple(f"novel_{i:02d}" for i in range(8)),
        ("capture_a", "capture_b", "capture_c", "capture_d"), ("visit_x", "visit_y"), "synthetic_training")
    people = {p: spec.known_sessions for p in spec.known_ids}
    people.update({p: spec.unknown_sessions for p in spec.unknown_ids})
    rows = tuple(SampleProvenance(p, s, c, i, 1000 + i * 250, 480, 270)
        for p, sessions in sorted(people.items()) for s in sessions for c in CONDITIONS for i in range(samples))
    y = np.array([spec.known_ids.index(r.person_id) if r.person_id in spec.known_ids else -1 for r in rows], dtype=np.int32)
    X = np.random.default_rng(713).normal(size=(len(rows), 1434)).astype(np.float32)
    X[:, 0] = y
    X[:, 1] = [spec.known_sessions.index(r.session_id) + 1 if label >= 0 else spec.unknown_sessions.index(r.session_id) + 1
               for r, label in zip(rows, y)]
    X[:, 2] = [r.sample_index for r in rows]
    X[:, 3] = [CONDITIONS.index(r.condition) for r in rows]
    X[:, 4:100] += y[:, None] * 2
    root = folder / "pose_processed"
    refs = []
    for start in range(0, len(rows), samples):
        p = rows[start]
        path = root / POSE_REPRESENTATION / p.source_relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if numeric:
            content = "\n".join(json.dumps(dict(schema_version=1, representation=POSE_REPRESENTATION, **asdict(r),
                source_relative_path=r.source_relative_path, landmark_count=478, feature_count=1434,
                features=X[i].round(4).tolist()), separators=(",", ":")) for i, r in enumerate(rows[start:start + samples], start)) + "\n"
        else:
            content = "SYNTHETIC_PAYLOAD_DO_NOT_READ\n"
        path.write_text(content, encoding="utf-8", newline="\n")
        refs.append(CatalogFile(p.person_id, p.session_id, p.condition,
            f"all_xyz_v1/{p.source_relative_path}", path.stat().st_size))
    metadata = dict(schema_version=1, completion_state="completed", participants=len(people), sessions=sum(map(len, people.values())),
        conditions=len(refs), total_samples_processed=len(rows), rejected_samples=0, error_count=0,
        normalization="PoseLandmarkNormalizer", normalization_parameters=POSE_NORMALIZATION,
        feature_extraction="LandmarkFeatureExtractor", inference_dimensions="per_source_sample")
    (root / POSE_REPRESENTATION / "processing_manifest.json").write_text(json.dumps(dict(metadata,
        representation=POSE_REPRESENTATION, condition_files=len(refs))), encoding="utf-8")
    (root / "processing_manifest.json").write_text(json.dumps(dict(metadata, representations=[POSE_REPRESENTATION],
        total_condition_files=len(refs), condition_files_per_representation=len(refs))), encoding="utf-8")
    catalog = TrainingCatalog("all_xyz_v1", "a" * 64, len(rows),
        tuple(ObservedSessions(p, sessions) for p, sessions in sorted(people.items())), tuple(refs))
    # Fold bodies are opaque authenticated provenance. The operational reader
    # consumes only population/catalog roles; it never plans or executes CV.
    raw = dict(specification=asdict(spec), catalog=asdict(catalog), outer_folds=[], generation={},
        specification_sha256=metadata_sha256(spec), catalog_sha256=metadata_sha256(catalog), schema_version=1,
        protocol="v3_nested_session_identity_cv", completion_state="catalog_validated")
    raw["protocol_sha256"] = metadata_sha256(raw)
    protocol_path, pose_path = folder / "splits" / "protocol.json", folder / "reports" / "pose.json"
    protocol_path.parent.mkdir(parents=True)
    protocol_path.write_text(json.dumps(raw), encoding="utf-8")
    protocol = read_training_protocol(protocol_path)
    data = (load_training_payloads(root, protocol) if numeric else TrainingData(protocol, X, y, rows,
        sha256(X.tobytes()).hexdigest(), representation=POSE_REPRESENTATION))
    pose_catalog = read_training_catalog(root)
    pose_path.parent.mkdir()
    pose_path.write_text(json.dumps(dict(schema_version=1, completion_state="completed", experiment="v3_pose_representation_comparison_v1",
        protocol_sha256=protocol.protocol_sha256, protocol_file_sha256=file_sha256(protocol_path), selected_development_representation=POSE_REPRESENTATION,
        normalization_definition_sha256=metadata_sha256(POSE_NORMALIZATION), results=[dict(representation=POSE_REPRESENTATION,
            catalog_sha256=metadata_sha256(pose_catalog), processing_manifest_sha256=pose_catalog.processing_manifest_sha256,
            payload_sha256=data.payload_sha256)])), encoding="utf-8")
    evidence = tuple(FrozenEvidenceReference(kind, protocol_path.name if kind == "protocol" else pose_path.name if kind == "pose_report" else kind + ".json",
        file_sha256(protocol_path) if kind == "protocol" else file_sha256(pose_path) if kind == "pose_report" else "a" * 64, 1, source)
        for kind, source in zip(EVIDENCE_KINDS, EVIDENCE_SOURCE_TYPES))
    config = FrozenV3OpenSetConfiguration(current_pose_normalization(), metadata_sha256(POSE_NORMALIZATION), current_implementation_fingerprints(),
        evidence, protocol.protocol_sha256, protocol.catalog_sha256, (DevelopmentMetricEvidence("pose_report", "mean_accuracy", .8),))
    config = replace(config, configuration_sha256=metadata_sha256(config))
    config_path = folder / "models" / "config.json"
    config_path.parent.mkdir()
    config_path.write_text(frozen_configuration_to_json(config), encoding="utf-8")
    provenance = V3TrainingProvenance(config_path.name, file_sha256(config_path), file_sha256(protocol_path), file_sha256(pose_path),
        metadata_sha256(pose_catalog), pose_catalog.processing_manifest_sha256, file_sha256(root / "processing_manifest.json"), data.payload_sha256)
    return data, config, provenance, root, config_path, protocol_path, pose_path


class RecordingDeployableRunner:
    def __init__(self):
        self.fit_count, self.events, self.model_calls = 0, [], []
        self.versions = EvaluationVersions(python_version(), np.__version__, scipy.__version__, sklearn.__version__)

    def fit(self, population, identities, methods=("mahalanobis_distance",)):
        assert np.all(population.y >= 0) and methods == ("mahalanobis_distance",)
        self.fit_count += 1
        sessions = tuple(sorted({r.session_id for r in population.samples}))
        self.events.append(("fit", sessions, tuple(sorted({r.person_id for r in population.samples})), len(population.y), methods))
        return SimpleNamespace(sessions=sessions, offset=float(population.X[:, 1].mean()) * .01,
            methods=methods, identity_count=identities)

    def infer(self, model, population):
        self.model_calls.append((self.fit_count, id(model), tuple(sorted({r.person_id for r in population.samples})),
            tuple(sorted({r.session_id for r in population.samples}))))
        known = population.X[:, 0] >= 0
        predicted = np.where(known, population.X[:, 0], 0).astype(np.int32)
        predicted[known & (population.X[:, 2] == 7)] = 1 - predicted[known & (population.X[:, 2] == 7)]
        return FrameScores(predicted, np.eye(model.identity_count)[predicted] * 3,
            (("mahalanobis_distance", np.where(known, -2., -5.) + model.offset),))

    @staticmethod
    def model_identity(model): return (id(model),)

    def fingerprints(self, model): return FittedV3Fingerprints(*(fitted_value_sha256(model) for _ in range(6)))

    @staticmethod
    def support_vector_count(model): return 12
