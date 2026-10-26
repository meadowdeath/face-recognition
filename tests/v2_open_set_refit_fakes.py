"""Small synthetic feature files and a synthetic frozen V2 config; never real data."""

from dataclasses import asdict
from hashlib import sha256
import json

import numpy as np

from face_recognition.application.rejection_metrics import evaluate_threshold
from face_recognition.application.temporal_windows import build_windows
from face_recognition.application.v2_study_data import known_references, study_folds
from face_recognition.domain.models.baseline_evaluation import ClassificationMetrics
from face_recognition.domain.models.identity_split import IdentityPartition
from face_recognition.domain.models.v2_open_set_config import FrozenV2OpenSetConfiguration, V2DevelopmentEvidence, V2ReportProvenance
from face_recognition.infrastructure.persistence.v2_open_set_config_repository import frozen_v2_configuration_to_json
from face_recognition.infrastructure.persistence.v2_study_loader import V2StudyLoader
from tests.unknown_rejection_fakes import rejection_fixture


def v2_refit_fixture(directory):
    processed, source, partition, source_path, partition_path = rejection_fixture(directory, samples=8, landmarks=478)
    generator = np.random.default_rng(703)
    known_ids, unknown_ids = source.specification.known_ids, source.specification.unknown_ids
    for path in sorted({processed / s.source_relative_path for s in source.samples}):
        records = [json.loads(line) for line in path.read_text().splitlines()]
        for record in records:
            person = record["person_id"]
            center = known_ids.index(person) * 3 if person in known_ids else (-6, 9, 12)[unknown_ids.index(person)]
            record["features"] = (generator.normal(0, .2, record["feature_count"]) + center).tolist()
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    known = known_references(source)
    unknown = tuple(s for s in source.samples if s.partition is IdentityPartition.UNKNOWN_POOL)
    metrics = evaluate_threshold(np.array([1.]), np.array([True]), np.array([0.]), .5)
    evidence = V2DevelopmentEvidence(ClassificationMetrics(.9, .9, .9, .9), .9, .1, metrics,
                                     len(known), len(unknown), len(known_ids), len(unknown_ids),
                                     len(build_windows(known, 7).windows), 3 * len(build_windows(unknown, 7).windows))
    config = FrozenV2OpenSetConfiguration(V2ReportProvenance("d" * 64, "a" * 64, partition.source_identity_split_sha256),
                                          evidence, study_folds(source))
    config_path = directory / "open_set_config_v2.json"
    content = frozen_v2_configuration_to_json(config).encode()
    config_path.write_bytes(content)
    return processed, source, partition, source_path, partition_path, config_path, config, sha256(content).hexdigest()


def load_v2_refit_fixture(fixture):
    processed, source, partition, _, _, _, config, _ = fixture
    return V2StudyLoader(processed).load_frozen_development(source, partition, config)
