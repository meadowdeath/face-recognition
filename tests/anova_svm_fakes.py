"""Small synthetic processed data supporting the real fixed 768-wide grid."""

import json
from pathlib import Path

import numpy as np

from face_recognition.domain.models.identity_split import IdentityPartition
from tests.training_fold_fakes import training_fixture


def tuning_fixture(directory: Path):
    processed, root, manifest, split = training_fixture(directory, landmarks=256, samples=3)
    generator = np.random.default_rng(742)
    for path in sorted({processed / row.source_relative_path for row in manifest.samples
                        if row.partition is IdentityPartition.KNOWN_TRAIN_POOL}):
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for row in records:
            values = generator.normal(size=row["feature_count"]).astype(np.float32)
            values[0] += manifest.specification.known_ids.index(row["person_id"]) * 8
            row["features"] = values.tolist()
        path.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
    return processed, root, manifest, split
