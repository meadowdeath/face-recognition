"""Explicit final-stage authorization; development/calibration payloads stay unopened."""

from dataclasses import asdict
import json

import numpy as np

from face_recognition.application.final_evaluation_data import FinalEvaluationError, FinalHoldoutData
from face_recognition.application.unknown_rejection_split import validate_unknown_split
from face_recognition.domain.models.identity_split import IdentityPartition
from face_recognition.infrastructure.persistence.development_dataset_loader import DevelopmentDatasetLoader


def validate_final_provenance(model, frozen, source, partition, development_report) -> None:
    """Bind final identity roles to the digest-bound pre-freeze development report."""
    validate_unknown_split(source, partition)
    if (development_report.sha256 != frozen.configuration.provenance.sha256
            or json.dumps(development_report.payload["unknown_partition"], sort_keys=True) != json.dumps(asdict(partition), sort_keys=True)
            or model.manifest["source_identity_split_sha256"] != partition.source_identity_split_sha256
            or source.specification.known_ids != model.parts.class_ids):
        raise FinalEvaluationError("Final split/model/development provenance mismatch")


class FinalHoldoutLoader(DevelopmentDatasetLoader):
    def load_final(self, source, partition, class_ids: tuple[str, ...], feature_count: int) -> FinalHoldoutData:
        validate_unknown_split(source, partition)
        if class_ids != source.specification.known_ids:
            raise FinalEvaluationError("Model classes do not match the final identity split")
        self._validate_catalog(source)  # Filenames/count metadata only.
        known = tuple(s for s in source.samples if s.partition is IdentityPartition.KNOWN_TEST)
        unknown = tuple(s for s in source.samples if s.partition is IdentityPartition.UNKNOWN_POOL
                        and s.person_id in partition.specification.holdout_ids)
        K = self._read_references(known, "final known holdout")
        U = self._read_references(unknown, "final unknown holdout")
        if K.shape[1] != feature_count or U.shape[1] != feature_count:
            raise FinalEvaluationError("Final features do not match fitted model dimensions")
        labels = {person: index for index, person in enumerate(class_ids)}
        y = np.fromiter((labels[s.person_id] for s in known), dtype=np.int32)
        return FinalHoldoutData(K, y, known, U, unknown, class_ids, source, partition)
