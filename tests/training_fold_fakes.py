"""Synthetic-only fixtures for development loading; no external dataset paths."""

from pathlib import Path

from face_recognition.application.create_identity_split import IdentitySplitPlanner
from face_recognition.domain.models.dataset import CONDITIONS
from face_recognition.domain.models.identity_split import IdentitySplitSpec
from face_recognition.infrastructure.persistence.feature_metadata_reader import FeatureMetadataReader
from face_recognition.infrastructure.persistence.identity_split_repository import write_identity_split
from tests.identity_split_fakes import write_feature_dataset


def training_fixture(directory: Path, *, training=("visit_a", "visit_b"), conditions=CONDITIONS[:2],
                     samples=2, landmarks=3, known=("known_beta", "known_alpha")):
    processed = directory / "processed"
    people = {person: (*training, "held_out") for person in known}
    people["reserved_identity"] = ("unknown_one", "unknown_two", "unknown_three")
    root = write_feature_dataset(processed, people, conditions, samples, landmarks)
    spec = IdentitySplitSpec(known, ("reserved_identity",), training, ("held_out",))
    # Fixture creation may scan all synthetic payloads. Training loading may not.
    manifest = IdentitySplitPlanner().plan(FeatureMetadataReader(processed), spec)
    split_path = directory / "split.json"
    write_identity_split(manifest, split_path, processed)
    return processed, root, manifest, split_path
