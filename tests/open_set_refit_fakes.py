"""Synthetic processed data plus a synthetic authoritative frozen configuration."""

from hashlib import sha256

from face_recognition.application.freeze_open_set_config import freeze_development_configuration
from face_recognition.infrastructure.persistence.open_set_config_repository import frozen_configuration_to_json, read_frozen_configuration
from tests.open_set_config_fakes import synthetic_development_report, synthetic_report_bytes
from tests.unknown_rejection_fakes import rejection_fixture, load_rejection_fixture


def refit_fixture(directory):
    processed, source, partition, source_path, partition_path = rejection_fixture(directory, samples=8, landmarks=478)
    config_path = directory / "frozen.json"
    config = freeze_development_configuration(synthetic_development_report(), sha256(synthetic_report_bytes()).hexdigest())
    config_path.write_text(frozen_configuration_to_json(config), encoding="utf-8")
    return processed, source, partition, source_path, partition_path, config_path


def load_refit_fixture(fixture):
    processed, source, partition, _, _, config_path = fixture
    known, unknown = load_rejection_fixture(processed, source, partition)
    return known, unknown, read_frozen_configuration(config_path)
