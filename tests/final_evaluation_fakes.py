"""Synthetic development-only fitting followed by separate synthetic final populations."""

from dataclasses import asdict
from hashlib import sha256
import json

from face_recognition.application.freeze_open_set_config import freeze_development_configuration
from face_recognition.application.refit_open_set_model import OpenSetModelRefitter
from face_recognition.domain.models.baseline_evaluation import EvaluationVersions
from face_recognition.domain.models.final_evaluation import FinalModelContext
from face_recognition.infrastructure.classifiers.sklearn_open_set_refit import SklearnOpenSetRefitRunner
from face_recognition.infrastructure.persistence.final_holdout_loader import FinalHoldoutLoader
from face_recognition.infrastructure.persistence.model_repository import load_open_set_model, model_fingerprints, persist_open_set_model
from face_recognition.infrastructure.persistence.open_set_config_repository import frozen_configuration_to_json, read_frozen_configuration
from tests.open_set_config_fakes import synthetic_development_report
from tests.open_set_refit_fakes import refit_fixture, load_refit_fixture


def final_fixture(directory):
    fixture = refit_fixture(directory)
    processed, source, partition, source_path, partition_path, config_path = fixture
    report = synthetic_development_report()
    report["unknown_partition"] = json.loads(json.dumps(asdict(partition)))
    report["identity_count"], report["label_order"] = source.known_identity_count, list(source.specification.known_ids)
    development_path = directory / "development.json"
    content = (json.dumps(report, indent=2) + "\n").encode("utf-8")
    development_path.write_bytes(content)
    config = freeze_development_configuration(report, sha256(content).hexdigest())
    config_path.write_text(frozen_configuration_to_json(config), encoding="utf-8")
    known, unknown, frozen = load_refit_fixture(fixture)
    fitted = OpenSetModelRefitter(SklearnOpenSetRefitRunner()).run(frozen.configuration, frozen.sha256, known, unknown)
    model_dir = directory / "model"
    persist_open_set_model(fitted, frozen, model_dir, (processed, source_path, partition_path, config_path))
    return fixture, development_path, model_dir


def load_final_fixture(bundle):
    fixture, _, model_dir = bundle
    processed, source, partition, _, _, config_path = fixture
    frozen = read_frozen_configuration(config_path)
    model = load_open_set_model(model_dir, frozen)
    context = FinalModelContext(frozen.configuration, frozen.sha256, model_fingerprints(model_dir), model.threshold,
                                model.manifest["calibration_analysis"]["roc_auc"],
                                model.manifest["calibration_analysis"]["calibration"]["metrics"]["binary_balanced_accuracy"],
                                EvaluationVersions(**model.manifest["versions"]))
    data = FinalHoldoutLoader(processed).load_final(source, partition, model.parts.class_ids, 1434)
    return data, model, context
