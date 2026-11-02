"""Synthetic-only deployable inputs and role-recording runner, no real data."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil

from face_recognition.application.create_v3_cv_protocol import metadata_sha256
from face_recognition.domain.models.processed_sample import POSE_REPRESENTATION
from face_recognition.domain.models.v3_deployable_model import FittedV3Fingerprints, V3TrainingProvenance
from face_recognition.domain.models.v3_open_set_config import (
    EVIDENCE_KINDS, EVIDENCE_SOURCE_TYPES, DevelopmentMetricEvidence, FrozenEvidenceReference, FrozenV3OpenSetConfiguration,
)
from face_recognition.infrastructure.features.pose_landmark_normalizer import POSE_NORMALIZATION
from face_recognition.infrastructure.persistence.artifact_integrity import file_sha256
from face_recognition.infrastructure.persistence.v3_cv_protocol_repository import read_v3_catalog, write_v3_cv_protocol
from face_recognition.infrastructure.persistence.v3_open_set_config import (
    current_implementation_fingerprints, current_pose_normalization, write_frozen_v3_configuration,
)
from face_recognition.infrastructure.persistence.v3_payload_loader import load_v3_payloads
from tests.v3_mahalanobis_fakes import RecordingSameModelRunner
from tests.v3_novelty_fakes import novelty_data, write_numeric_payloads


def deployable_fixture(folder, *, numeric=False):
    folder = Path(folder)
    data, baseline, _ = novelty_data(folder, samples=8)
    protocol = write_numeric_payloads(data, baseline, data.protocol.specification) if numeric else data.protocol
    root = folder / "pose_processed"
    shutil.copytree(baseline / "all_xyz_v1", root / POSE_REPRESENTATION)
    header = root / POSE_REPRESENTATION / "processing_manifest.json"
    metadata = json.loads(header.read_text(encoding="utf-8"))
    metadata.update(representation=POSE_REPRESENTATION, normalization="PoseLandmarkNormalizer", normalization_parameters=POSE_NORMALIZATION,
        feature_extraction="LandmarkFeatureExtractor", inference_dimensions="per_source_sample")
    header.write_text(json.dumps(metadata), encoding="utf-8")
    run = {k: v for k, v in metadata.items() if k not in ("representation", "condition_files")}
    run.update(representations=[POSE_REPRESENTATION], total_condition_files=metadata["condition_files"], condition_files_per_representation=metadata["condition_files"])
    (root / "processing_manifest.json").write_text(json.dumps(run), encoding="utf-8")
    if numeric:
        for path in (root / POSE_REPRESENTATION).rglob("*.jsonl"):
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            for row in rows: row["representation"] = POSE_REPRESENTATION
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        data = load_v3_payloads(root, protocol, representation=POSE_REPRESENTATION)
    else:
        data = replace(data, representation=POSE_REPRESENTATION)
    protocol_path, pose_path = folder / "splits" / "protocol.json", folder / "reports" / "pose.json"
    write_v3_cv_protocol(protocol, protocol_path, ())
    catalog = read_v3_catalog(root, representation=POSE_REPRESENTATION)
    pose_path.parent.mkdir()
    pose_path.write_text(json.dumps(dict(schema_version=1, completion_state="completed", experiment="v3_pose_representation_comparison_v1",
        protocol_sha256=protocol.protocol_sha256, protocol_file_sha256=file_sha256(protocol_path), selected_development_representation=POSE_REPRESENTATION,
        normalization_definition_sha256=metadata_sha256(POSE_NORMALIZATION), results=[dict(representation=POSE_REPRESENTATION,
            catalog_sha256=metadata_sha256(catalog), processing_manifest_sha256=catalog.processing_manifest_sha256, payload_sha256=data.payload_sha256)])), encoding="utf-8")
    refs = tuple(FrozenEvidenceReference(kind, protocol_path.name if kind == "protocol" else pose_path.name if kind == "pose_report" else kind + ".json",
        file_sha256(protocol_path) if kind == "protocol" else file_sha256(pose_path) if kind == "pose_report" else "a" * 64,
        1, source) for kind, source in zip(EVIDENCE_KINDS, EVIDENCE_SOURCE_TYPES))
    config = FrozenV3OpenSetConfiguration(current_pose_normalization(), metadata_sha256(POSE_NORMALIZATION), current_implementation_fingerprints(),
        refs, protocol.protocol_sha256, protocol.catalog_sha256, (DevelopmentMetricEvidence("pose_report", "mean_accuracy", .8),))
    config = replace(config, configuration_sha256=metadata_sha256(config))
    config_path = folder / "models" / "config.json"
    write_frozen_v3_configuration(config, config_path, ())
    provenance = V3TrainingProvenance(config_path.name, file_sha256(config_path), file_sha256(protocol_path), file_sha256(pose_path),
        metadata_sha256(catalog), catalog.processing_manifest_sha256, file_sha256(root / "processing_manifest.json"), data.payload_sha256)
    return data, config, provenance, root, config_path, protocol_path, pose_path


class RecordingDeployableRunner(RecordingSameModelRunner):
    def fit(self, population, identities, methods=("mahalanobis_distance",)):
        model = super().fit(population, identities, methods)
        model.identity_count = identities
        return model

    def fingerprints(self, model):
        digest = self.model_state_sha256(model)
        return FittedV3Fingerprints(*(digest for _ in range(6)))

    @staticmethod
    def support_vector_count(model):
        return 12
