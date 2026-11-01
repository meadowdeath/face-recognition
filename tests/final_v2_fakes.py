"""Entirely synthetic development/model/final cohorts in one temporary directory."""

from hashlib import sha256
import json

import numpy as np

from face_recognition.application.create_identity_split import IdentitySplitPlanner
from face_recognition.application.freeze_v2_open_set_config import freeze_v2_configuration
from face_recognition.application.refit_v2_open_set_model import V2OpenSetRefitter
from face_recognition.application.unknown_rejection_split import plan_unknown_split
from face_recognition.domain.models.identity_split import IdentitySplitSpec
from face_recognition.domain.models.unknown_rejection import UnknownSplitSpec
from face_recognition.infrastructure.classifiers.sklearn_v2_open_set_refit import SklearnV2OpenSetRefitRunner
from face_recognition.infrastructure.persistence.feature_metadata_reader import FeatureMetadataReader
from face_recognition.infrastructure.persistence.identity_split_repository import write_identity_split
from face_recognition.infrastructure.persistence.unknown_rejection_repository import write_unknown_split
from face_recognition.infrastructure.persistence.v2_model_repository import persist_v2_open_set_model
from face_recognition.infrastructure.persistence.v2_open_set_config_repository import frozen_v2_configuration_to_json
from face_recognition.infrastructure.persistence.v2_study_loader import V2StudyLoader
from tests.identity_split_fakes import write_feature_dataset
from tests.v2_open_set_config_fakes import synthetic_v2_report


def final_v2_fixture(directory):
    known = ("synthetic_a", "synthetic_b")
    old_unknown = ("old_unknown_a", "old_unknown_b")
    development = directory / "development"
    people = {p: ("visit_a", "visit_b", "visit_c") for p in known}
    people.update({p: ("old_visit",) for p in old_unknown})
    write_feature_dataset(development, people, conditions=("center",), samples=14, landmarks=478)
    source = IdentitySplitPlanner().plan(FeatureMetadataReader(development), IdentitySplitSpec(known, old_unknown, ("visit_a", "visit_b"), ("visit_c",)))
    partition = plan_unknown_split(source, UnknownSplitSpec((old_unknown[0],), (old_unknown[1],)))
    source_path, unknown_path = directory / "identity.json", directory / "unknown.json"
    write_identity_split(source, source_path, development)
    write_unknown_split(partition, source, unknown_path, source_path)
    rng = np.random.default_rng(712)
    def populate(root):
        for path in sorted(root.rglob("*.jsonl")):
            records = [json.loads(line) for line in path.read_text().splitlines()]
            for row in records:
                center = (-3 if row["person_id"] == known[0] else 3) if row["person_id"] in known else 10
                row["features"] = rng.normal(center, .15, row["feature_count"]).tolist()
            path.write_text("\n".join(json.dumps(row) for row in records) + "\n")
    populate(development)
    report = synthetic_v2_report()
    report["v1_baseline"]["identity_split_sha256"] = partition.source_identity_split_sha256
    development_report = directory / "v2_diagnostic.json"
    content = json.dumps(report).encode()
    development_report.write_bytes(content)
    config = freeze_v2_configuration(report, sha256(content).hexdigest())
    config_path = directory / "frozen_v2.json"
    config_path.write_text(frozen_v2_configuration_to_json(config))
    digest = sha256(config_path.read_bytes()).hexdigest()
    data = V2StudyLoader(development).load_frozen_development(source, partition, config)
    fitted = V2OpenSetRefitter(SklearnV2OpenSetRefitRunner()).run(config, digest, data)
    model_dir = directory / "model_v2"
    persist_v2_open_set_model(fitted, config, digest, model_dir, (development, source_path, unknown_path, config_path))
    final_root = directory / "independent_final"
    final_people = {p: ("independent_visit",) for p in known}
    final_people.update(new_unknown_a=("new_visit",), new_unknown_b=("new_visit", "other_visit"))
    write_feature_dataset(final_root, final_people, conditions=("center", "translation_left"), samples=8, landmarks=478)
    populate(final_root)
    return final_root, source_path, unknown_path, model_dir, config_path, development_report
