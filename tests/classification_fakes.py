"""Small in-memory synthetic classifier/feature-transform datasets."""
from dataclasses import replace
import numpy as np
from face_recognition.application.training_folds import DevelopmentDataset
from face_recognition.domain.models.identity_split import IdentityPartition, SplitSampleReference
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

def reduction_data(feature_count=9):
    data = development_data(feature_count=feature_count, samples_per_identity=8)
    X = np.random.default_rng(471).normal(size=data.X.shape).astype(np.float32)
    X[:, 0] += data.y * 8
    return replace(data, X=X)
