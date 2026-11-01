"""In-memory numeric populations and a partition-recording deterministic runner."""

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from face_recognition.application.create_v3_cv_protocol import create_v3_cv_protocol
from face_recognition.application.v3_novelty_data import FrameScores, V3PayloadData
from face_recognition.domain.models.dataset import CONDITIONS
from face_recognition.domain.models.processed_sample import SampleProvenance
from face_recognition.infrastructure.classifiers.sklearn_classification import SklearnClassifierRunner, classification_measurements
from face_recognition.infrastructure.persistence.v3_cv_protocol_repository import read_v3_catalog
from tests.v3_protocol_fakes import v3_catalog_fixture


def novelty_data(directory: Path, samples=8):
    processed, spec_path, spec = v3_catalog_fixture(directory, known_count=2)
    header = processed / spec.representation / "processing_manifest.json"
    metadata = json.loads(header.read_text(encoding="utf-8"))
    metadata["total_samples_processed"] = metadata["condition_files"] * samples
    header.write_text(json.dumps(metadata), encoding="utf-8")
    protocol = create_v3_cv_protocol(spec, read_v3_catalog(processed))
    rows, labels = [], []
    for reference in protocol.catalog.condition_files:
        label = spec.known_ids.index(reference.person_id) if reference.person_id in spec.known_ids else -1
        for index in range(samples):
            rows.append(SampleProvenance(reference.person_id, reference.session_id, reference.condition, index, 1000 + index * 250, 480, 270))
            labels.append(label)
    X = np.random.default_rng(713).normal(size=(len(rows), 1434)).astype(np.float32)
    y = np.asarray(labels, dtype=np.int32)
    X[:, 0] = y
    X[:, 1] = [spec.known_sessions.index(r.session_id) + 1 if label >= 0 else spec.unknown_sessions.index(r.session_id) + 1
               for r, label in zip(rows, labels)]
    X[:, 2] = [r.sample_index for r in rows]
    X[:, 3] = [CONDITIONS.index(r.condition) for r in rows]
    X[:, 4:100] += y[:, None] * 2
    return V3PayloadData(protocol, X, y, tuple(rows), sha256(X.tobytes()).hexdigest()), processed, spec_path


def write_numeric_payloads(data, processed, spec):
    """Only needed by loader/CLI tests; other tests avoid large JSON fixtures."""
    offset = 0
    for reference in data.protocol.catalog.condition_files:
        records = []
        while offset < len(data.samples):
            p = data.samples[offset]
            if (p.person_id, p.session_id, p.condition) != (reference.person_id, reference.session_id, reference.condition):
                break
            records.append(json.dumps({"schema_version": 1, "representation": "all_xyz_v1", **asdict(p),
                 "source_relative_path": p.source_relative_path, "landmark_count": 478, "feature_count": 1434,
                 "features": data.X[offset].round(4).tolist()}, separators=(",", ":")))
            offset += 1
        (processed / reference.source_relative_path).write_text("\n".join(records) + "\n", encoding="utf-8")
    return create_v3_cv_protocol(spec, read_v3_catalog(processed))


class RecordingNoveltyRunner:
    def __init__(self, final_shift=0.):
        self.events = []
        self.final_shift = final_shift
        self.versions = SklearnClassifierRunner().versions

    @staticmethod
    def classification_metrics(y, predicted, identities):
        return classification_measurements(y, predicted, identities)[0]

    def fit(self, population, identities, methods=("svm_top1", "svm_margin", "centroid_distance", "mahalanobis_distance")):
        assert np.all(population.y >= 0), "UNKNOWN used to fit"
        sessions = tuple(sorted({p.session_id for p in population.samples}))
        self.events.append(("fit", sessions, tuple(sorted({p.person_id for p in population.samples})), len(population.samples), methods))
        offset = float(population.X[:, 1].mean()) * .01 + (self.final_shift if len(sessions) == 3 else 0.)
        return SimpleNamespace(sessions=sessions, offset=offset, methods=methods)

    def infer(self, model, population):
        self.events.append(("infer", model.sessions, tuple(sorted({p.session_id for p in population.samples})),
                            tuple(sorted({p.person_id for p in population.samples})), len(population.samples)))
        known = population.X[:, 0] >= 0
        predicted = np.where(known, population.X[:, 0], 0).astype(np.int32)
        # Trailing errors remain part of closed-set frame evaluation.
        predicted[known & (population.X[:, 2] == 7)] = 1 - predicted[known & (population.X[:, 2] == 7)]
        decisions = np.eye(2)[predicted] * 3
        confidence = {"svm_top1": np.where(known, 3., 1.), "svm_margin": np.where(known, 2., 1.),
                      "centroid_distance": np.where(known, -1., -4.), "mahalanobis_distance": np.where(known, -2., -5.)}
        return FrameScores(predicted, decisions, tuple((method, confidence[method] + model.offset) for method in model.methods))
