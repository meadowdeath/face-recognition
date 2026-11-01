"""Catalog-only independent final manifest creation/validation and payload loading."""

from dataclasses import MISSING, asdict, dataclass
from hashlib import sha256
import json

import numpy as np

from face_recognition.application.final_evaluation_data import FinalEvaluationError
from face_recognition.application.final_v2_holdout import FinalV2Data, validate_v2_final_provenance, validate_v2_final_roles
from face_recognition.domain.models.final_v2_evaluation import FinalV2Condition, FinalV2HoldoutManifest, FinalV2Sample, FinalV2Session
from face_recognition.infrastructure.persistence.baseline_evaluation_report import write_aggregate_report_json
from face_recognition.infrastructure.persistence.development_dataset_loader import DevelopmentDatasetLoader
from face_recognition.infrastructure.persistence.feature_metadata_reader import FeatureMetadataReader
from face_recognition.infrastructure.persistence.identity_split_repository import read_identity_split
from face_recognition.infrastructure.persistence.open_set_config_repository import read_aggregate_development_report
from face_recognition.infrastructure.persistence.unknown_rejection_repository import read_unknown_split
from face_recognition.infrastructure.persistence.v2_model_repository import read_v2_model_metadata
from face_recognition.infrastructure.persistence.v2_open_set_config_repository import frozen_v2_configuration_from_json


@dataclass(frozen=True)
class V2FinalInputs:
    config: object
    config_sha256: str
    source: object
    partition: object
    unknown_split_sha256: str
    model_metadata: dict
    model_fingerprints: tuple


def read_v2_final_inputs(identity_split, unknown_split, frozen_config, development_report, model_dir):
    """No final catalog or development feature files opened; model stays serialized."""
    frozen = read_aggregate_development_report(frozen_config)
    config = frozen_v2_configuration_from_json(frozen.payload)
    source = read_identity_split(identity_split)
    partition = read_unknown_split(unknown_split, source)
    development = read_aggregate_development_report(development_report)
    unknown_sha = sha256(unknown_split.read_bytes()).hexdigest()
    metadata, fingerprints = read_v2_model_metadata(model_dir, config, frozen.sha256)
    validate_v2_final_provenance(config, frozen.sha256, source, partition, development, metadata, unknown_sha)
    return V2FinalInputs(config, frozen.sha256, source, partition, unknown_sha, metadata, fingerprints)


def final_v2_manifest_to_json(manifest):
    return json.dumps(asdict(manifest), allow_nan=False, sort_keys=True, indent=2) + "\n"


def read_final_v2_manifest(path):
    document = read_aggregate_development_report(path)
    try:
        data = document.payload
        kwargs = {name: data[name] for name, f in FinalV2HoldoutManifest.__dataclass_fields__.items() if f.default is MISSING}
        kwargs["known_sessions"] = tuple(FinalV2Session(**v) for v in data["known_sessions"])
        kwargs["unknown_sessions"] = tuple(FinalV2Session(**v) for v in data["unknown_sessions"])
        kwargs["condition_files"] = tuple(FinalV2Condition(**v) for v in data["condition_files"])
        result = FinalV2HoldoutManifest(**kwargs)
        if json.dumps(data, sort_keys=True, allow_nan=False) != json.dumps(json.loads(final_v2_manifest_to_json(result)), sort_keys=True):
            raise ValueError
        return result, document.sha256
    except Exception:
        raise FinalEvaluationError("Invalid or incomplete independent V2 final manifest") from None


class FinalV2HoldoutLoader(DevelopmentDatasetLoader):
    def catalog_manifest(self, inputs):
        """Reads only names and processing_manifest.json; NEVER opens JSONL."""
        keys = self.discover_conditions()
        classes = tuple(inputs.model_metadata["class_ids"])
        pairs = sorted({FinalV2Session(p, s) for p, s, _ in keys})
        manifest = FinalV2HoldoutManifest(inputs.config_sha256, inputs.config.provenance.sha256,
                    inputs.config.provenance.original_identity_split_sha256, inputs.unknown_split_sha256,
                    dict(inputs.model_fingerprints)["manifest.json"], self._processing_digest(),
                    tuple(row for row in pairs if row.person_id in classes),
                    tuple(row for row in pairs if row.person_id not in classes),
                    tuple(FinalV2Condition(*k) for k in keys), self.expected_sample_count)
        validate_v2_final_roles(manifest, inputs.source, classes)
        return manifest

    def _processing_digest(self):
        document = read_aggregate_development_report(self._checked(self.root / "processing_manifest.json"))
        if document.payload != self._manifest or self._read_manifest() != self._manifest:
            raise FinalEvaluationError("Final processing manifest changed during authorization")
        return document.sha256

    def authorize(self, manifest, inputs):
        validate_v2_final_roles(manifest, inputs.source, tuple(inputs.model_metadata["class_ids"]))
        if manifest != self.catalog_manifest(inputs):
            raise FinalEvaluationError("Final V2 manifest/catalog/provenance mismatch; no extra or missing session/file is allowed")

    def load_final(self, manifest, inputs):
        self.authorize(manifest, inputs)  # All role/catalog/provenance checks precede the first JSONL read.
        known_pairs = set(manifest.known_sessions)
        known, unknown = [], []
        # Reuse shared selective decoding, ordering, finite-value and dimension validation.
        # Metadata and features share JSONL records: payload checks happen HERE, not in catalog-only creation.
        for metadata in FeatureMetadataReader(self.processed_root).iter_metadata():
            p = metadata.provenance
            if metadata.feature_count != manifest.original_feature_count:
                raise FinalEvaluationError("Final V2 source feature dimension differs from frozen model")
            row = FinalV2Sample(p.person_id, p.session_id, p.condition, p.sample_index,
                               manifest.representation + "/" + p.source_relative_path)
            (known if FinalV2Session(p.person_id, p.session_id) in known_pairs else unknown).append(row)
        K = self._read_references(tuple(known), "independent V2 final KNOWN")
        U = self._read_references(tuple(unknown), "independent V2 final UNKNOWN")
        if self._processing_digest() != manifest.processing_manifest_sha256:
            raise FinalEvaluationError("Final V2 processing metadata changed during loading")
        classes = tuple(inputs.model_metadata["class_ids"])
        labels = {p: i for i, p in enumerate(classes)}
        y = np.fromiter((labels[s.person_id] for s in known), dtype=np.int32)
        return FinalV2Data(K, y, tuple(known), U, tuple(unknown), classes, manifest, inputs.source)


def write_final_v2_manifest(manifest, output, inputs):
    write_aggregate_report_json(final_v2_manifest_to_json(manifest), output, inputs)
