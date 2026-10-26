"""Read current V1-source data for explicitly diagnostic V2 use; reject future catalogs."""

from math import isfinite

import numpy as np

from face_recognition.application.unknown_rejection_split import validate_unknown_split
from face_recognition.application.v2_study_data import V2StudyData, V2StudyError, known_references, study_folds
from face_recognition.domain.models.identity_split import IdentityPartition
from face_recognition.domain.models.v2_diagnostics import V1Reference
from face_recognition.infrastructure.persistence.development_dataset_loader import DevelopmentDatasetLoader
from face_recognition.infrastructure.persistence.open_set_config_repository import read_aggregate_development_report


def read_v1_reference(path, source, partition) -> V1Reference:
    validate_unknown_split(source, partition)
    try:
        document = read_aggregate_development_report(path)
        r = document.payload
        if (type(r["schema_version"]) is not int or r["schema_version"] != 1 or r["experiment"] != "final_open_set_evaluation_v1"
                or r["status"] != "final_holdout_evaluation_completed" or r["model_artifacts_unchanged"] is not True
                or r["fitting_performed"] is not False or r["threshold_calibration_performed"] is not False
                or r["known_test_payloads_accessed"] is not True or r["unknown_holdout_payloads_accessed"] is not True
                or r["source_identity_split_sha256"] != partition.source_identity_split_sha256
                or tuple(r["closed_set_known_frames"]["label_order"]) != source.specification.known_ids):
            raise ValueError
        closed, temporal, metrics = r["closed_set_known_frames"]["metrics"], r["temporal_novelty_diagnostics"], r["operational_metrics"]
        values = (closed["accuracy"], closed["macro_f1"], temporal["roc_auc"], temporal["equal_error_point"]["eer"],
                  metrics["unknown_false_acceptance_rate"], metrics["known_rejection_rate"],
                  metrics["known_correct_identification_rate"], metrics["open_set_exact_accuracy"])
        if any(type(v) not in (int, float) or not isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError
        return V1Reference(document.sha256, partition.source_identity_split_sha256, *values)
    except Exception:
        raise V2StudyError("Invalid/incomplete V1 final reference or changed original split; no future data may be promoted") from None


class V2StudyLoader(DevelopmentDatasetLoader):
    def load_study(self, source, partition, baseline: V1Reference):
        validate_unknown_split(source, partition)
        study_folds(source)
        if baseline.identity_split_sha256 != partition.source_identity_split_sha256:
            raise V2StudyError("V2 source differs from the completed V1 experiment")
        self._validate_catalog(source)  # Future sessions/IDs fail before any feature file opens.
        known = known_references(source)
        unknown = tuple(s for s in source.samples if s.partition is IdentityPartition.UNKNOWN_POOL)
        K, U = self._read_references(known, "V2 known development"), self._read_references(unknown, "V2 unknown development")
        labels = {p: i for i, p in enumerate(source.specification.known_ids)}
        return V2StudyData(K, np.fromiter((labels[s.person_id] for s in known), dtype=np.int32), known, U, unknown, source, partition)
