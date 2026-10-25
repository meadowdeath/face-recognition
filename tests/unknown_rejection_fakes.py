"""Synthetic known and unknown identities only; no external real dataset paths."""

import json
from pathlib import Path

import numpy as np

from face_recognition.application.create_identity_split import IdentitySplitPlanner
from face_recognition.application.unknown_rejection_split import plan_unknown_split
from face_recognition.domain.models.identity_split import IdentityPartition, IdentitySplitSpec
from face_recognition.domain.models.unknown_rejection import UnknownSplitSpec
from face_recognition.infrastructure.persistence.development_dataset_loader import DevelopmentDatasetLoader
from face_recognition.infrastructure.persistence.feature_metadata_reader import FeatureMetadataReader
from face_recognition.infrastructure.persistence.identity_split_repository import write_identity_split
from face_recognition.infrastructure.persistence.unknown_calibration_loader import UnknownCalibrationLoader
from face_recognition.infrastructure.persistence.unknown_rejection_repository import write_unknown_split
from tests.identity_split_fakes import write_feature_dataset


def rejection_fixture(directory: Path):
    processed = directory / "processed"
    known_ids = ("known_a", "known_b", "known_c")
    unknown_ids = ("unknown_cal", "unknown_hold_a", "unknown_hold_b")
    people = {person: ("visit_a", "visit_b", "reserved_visit") for person in known_ids}
    people.update(unknown_cal=("u1", "u2"), unknown_hold_a=("u1", "u2"), unknown_hold_b=("one",))
    root = write_feature_dataset(processed, people, samples=3, landmarks=171)
    source = IdentitySplitPlanner().plan(FeatureMetadataReader(processed),
                                          IdentitySplitSpec(known_ids, unknown_ids, ("visit_a", "visit_b"), ("reserved_visit",)))
    partition = plan_unknown_split(source, UnknownSplitSpec(("unknown_cal",), ("unknown_hold_a", "unknown_hold_b")))
    generator = np.random.default_rng(492)
    # Populate only permitted files; protected payloads can be corrupted by tests.
    allowed = {processed / row.source_relative_path for row in source.samples
               if row.partition is IdentityPartition.KNOWN_TRAIN_POOL or row.person_id == "unknown_cal"}
    for path in sorted(allowed):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for row in rows:
            features = generator.normal(size=row["feature_count"]).astype(np.float32)
            if row["person_id"] in known_ids:
                features[:3] += known_ids.index(row["person_id"]) * 5
            row["features"] = features.tolist()
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    source_path, partition_path = directory / "identity.json", directory / "unknown.json"
    write_identity_split(source, source_path, processed)
    write_unknown_split(partition, source, partition_path, source_path)
    return processed, source, partition, source_path, partition_path


def load_rejection_fixture(processed, source, partition):
    known = DevelopmentDatasetLoader(processed).load(source)
    unknown = UnknownCalibrationLoader(processed).load_calibration(source, partition, known.X.shape[1])
    return known, unknown
