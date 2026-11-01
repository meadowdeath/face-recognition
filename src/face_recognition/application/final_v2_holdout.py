"""Role independence and strict logical final manifest validation, no feature access."""

from dataclasses import dataclass, field
import re

import numpy as np

from face_recognition.application.final_evaluation_data import FinalEvaluationError
from face_recognition.application.unknown_rejection_split import validate_unknown_split
from face_recognition.domain.models.dataset import CONDITIONS, validate_identifier
from face_recognition.domain.models.final_v2_evaluation import FinalV2HoldoutManifest, FinalV2Session


def validate_v2_final_roles(manifest, source, class_ids):
    try:
        if (manifest.schema_version != 1 or type(manifest.schema_version) is not int
                or manifest.representation != "all_xyz_v1" or manifest.original_feature_count != 1434
                or type(manifest.original_feature_count) is not int or manifest.completion_state != "catalog_validated"
                or type(manifest.expected_total_samples) is not int or manifest.expected_total_samples < 1):
            raise ValueError
        for digest in (manifest.frozen_config_sha256, manifest.development_report_sha256, manifest.development_identity_split_sha256,
                       manifest.development_unknown_split_sha256, manifest.model_manifest_sha256, manifest.processing_manifest_sha256):
            if not isinstance(digest, str) or re.fullmatch("[0-9a-f]{64}", digest) is None:
                raise ValueError
        for population in (manifest.known_sessions, manifest.unknown_sessions):
            if not population or tuple(sorted(set(population))) != population:
                raise ValueError
            for row in population:
                validate_identifier(row.person_id)
                validate_identifier(row.session_id)
        known = {s.person_id for s in manifest.known_sessions}
        unknown = {s.person_id for s in manifest.unknown_sessions}
        if known != set(class_ids) or known & unknown or tuple(class_ids) != source.specification.known_ids:
            raise FinalEvaluationError("Final KNOWN roles must exactly match frozen classes; final roles must be disjoint")
        development_ids = set(source.specification.known_ids) | set(source.specification.unknown_ids)
        development_sessions = {(s.person_id, s.session_id) for s in source.session_assignments}
        if unknown & development_ids:
            raise FinalEvaluationError("Final UNKNOWN identity already participated in development")
        if any((s.person_id, s.session_id) in development_sessions for s in manifest.known_sessions):
            raise FinalEvaluationError("Final KNOWN session already participated in development")
        pairs = set(manifest.known_sessions + manifest.unknown_sessions)
        keys = []
        for row in manifest.condition_files:
            validate_identifier(row.person_id)
            validate_identifier(row.session_id)
            if row.condition not in CONDITIONS or FinalV2Session(row.person_id, row.session_id) not in pairs:
                raise ValueError
            keys.append((row.person_id, row.session_id, row.condition))
        if (keys != sorted(set(keys), key=lambda k: (k[0], k[1], CONDITIONS.index(k[2])))
                or {FinalV2Session(k[0], k[1]) for k in keys} != pairs):
            raise ValueError
    except FinalEvaluationError:
        raise
    except Exception:
        raise FinalEvaluationError("Malformed final V2 role/session/condition metadata") from None


def validate_v2_final_provenance(config, config_sha, source, partition, development, model_metadata, unknown_split_sha):
    from face_recognition.application.freeze_v2_open_set_config import freeze_v2_configuration
    validate_unknown_split(source, partition)
    if (freeze_v2_configuration(development.payload, development.sha256) != config
            or development.sha256 != config.provenance.sha256
            or partition.source_identity_split_sha256 != config.provenance.original_identity_split_sha256
            or model_metadata["frozen_config_sha256"] != config_sha
            or model_metadata["diagnostic_report_sha256"] != development.sha256
            or model_metadata["source_identity_split_sha256"] != partition.source_identity_split_sha256
            or tuple(model_metadata["class_ids"]) != source.specification.known_ids
            or re.fullmatch("[0-9a-f]{64}", unknown_split_sha) is None):
        raise FinalEvaluationError("V2 model/configuration/development split/report provenance mismatch")


@dataclass(frozen=True, eq=False)
class FinalV2Data:
    known_X: np.ndarray = field(repr=False)
    known_y: np.ndarray = field(repr=False)
    known_samples: tuple = field(repr=False)
    unknown_X: np.ndarray = field(repr=False)
    unknown_samples: tuple = field(repr=False)
    class_ids: tuple[str, ...]
    manifest: FinalV2HoldoutManifest = field(repr=False)
    development_source: object = field(repr=False)

    def __post_init__(self):
        validate_v2_final_roles(self.manifest, self.development_source, self.class_ids)
        all_rows = self.known_samples + self.unknown_samples
        if (len(all_rows) != self.manifest.expected_total_samples or len({(s.person_id, s.session_id, s.condition, s.sample_index) for s in all_rows}) != len(all_rows)):
            raise FinalEvaluationError("Final V2 sample coverage/counts are inconsistent")
        for X, rows, sessions in ((self.known_X, self.known_samples, self.manifest.known_sessions),
                                  (self.unknown_X, self.unknown_samples, self.manifest.unknown_sessions)):
            if (not rows or X.dtype != np.float32 or X.shape != (len(rows), self.manifest.original_feature_count)
                    or not np.isfinite(X).all() or {FinalV2Session(s.person_id, s.session_id) for s in rows} != set(sessions)):
                raise FinalEvaluationError("Invalid final V2 sample matrix or session role")
            if (any(type(s.sample_index) is not int or s.sample_index < 0 or s.source_relative_path !=
                    f"all_xyz_v1/{s.person_id}/{s.session_id}/{s.condition}.jsonl" for s in rows)
                    or rows != tuple(sorted(rows, key=lambda s: (s.person_id, s.session_id, CONDITIONS.index(s.condition), s.sample_index)))):
                raise FinalEvaluationError("Final V2 row ordering/reference provenance is inconsistent")
            X.setflags(write=False)
        labels = {p: i for i, p in enumerate(self.class_ids)}
        if self.known_y.dtype != np.int32 or not np.array_equal(self.known_y, [labels[s.person_id] for s in self.known_samples]):
            raise FinalEvaluationError("Final V2 known class mapping is inconsistent")
        if {(s.person_id, s.session_id, s.condition) for s in all_rows} != {(s.person_id, s.session_id, s.condition) for s in self.manifest.condition_files}:
            raise FinalEvaluationError("Final V2 condition coverage is inconsistent")
        self.known_y.setflags(write=False)
