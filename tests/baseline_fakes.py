"""Small synthetic full-coordinate data; no real biometric files or values."""

import json
from pathlib import Path

import numpy as np

from face_recognition.application.training_folds import DevelopmentDataset
from face_recognition.domain.models.identity_split import IdentityPartition, SplitSampleReference
from tests.training_fold_fakes import training_fixture


def development_data(feature_count=9, samples_per_identity=6) -> DevelopmentDataset:
    identities, sessions = ("known_alpha", "known_beta"), ("visit_a", "visit_b")
    rows, labels, metadata = [], [], []
    for session_index, session in enumerate(sessions):
        for label, person in enumerate(identities):
            for index in range(samples_per_identity):
                rows.append([(-4 if label == 0 else 4) + session_index * 0.2 + index * 0.01 + column * 0.001
                             for column in range(feature_count)])
                labels.append(label)
                metadata.append(SplitSampleReference(person, session, "center", index, IdentityPartition.KNOWN_TRAIN_POOL,
                                                      f"all_xyz_v1/{person}/{session}/center.jsonl"))
    return DevelopmentDataset(np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=np.int32),
                              tuple(metadata), identities, sessions)


def baseline_fixture(directory: Path):
    processed, root, manifest, split = training_fixture(directory, samples=3)
    # Only synthetic development files receive class-separated features. Protected
    # files can subsequently be made unreadable to prove they are never consumed.
    paths = {processed / row.source_relative_path for row in manifest.samples
             if row.partition is IdentityPartition.KNOWN_TRAIN_POOL}
    for path in paths:
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for record in records:
            label = manifest.specification.known_ids.index(record["person_id"])
            record["features"] = [(-4 if label == 0 else 4) + record["sample_index"] * 0.01 + column * 0.001
                                  for column in range(record["feature_count"])]
        path.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
    return processed, root, manifest, split
