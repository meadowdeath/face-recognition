"""Authorize unknown calibration identities before opening any feature file."""

from face_recognition.application.rejection_data import UnknownCalibrationData
from face_recognition.application.unknown_rejection_split import validate_unknown_split
from face_recognition.domain.models.identity_split import IdentityPartition, IdentitySplitManifest
from face_recognition.domain.models.unknown_rejection import UnknownRejectionSplit
from face_recognition.infrastructure.persistence.development_dataset_loader import DevelopmentDatasetLoader, DevelopmentDatasetReadError


class UnknownCalibrationLoader(DevelopmentDatasetLoader):
    def load_calibration(self, source: IdentitySplitManifest, partition: UnknownRejectionSplit,
                         feature_count: int) -> UnknownCalibrationData:
        validate_unknown_split(source, partition)
        self._validate_catalog(source)  # Metadata/file names only, including protected partitions.
        references = tuple(row for row in source.samples if row.partition is IdentityPartition.UNKNOWN_POOL
                           and row.person_id in partition.specification.calibration_ids)
        X = self._read_references(references, "unknown calibration")
        if X.shape[1] != feature_count:
            raise DevelopmentDatasetReadError("Known and unknown calibration feature dimensions differ")
        return UnknownCalibrationData(X, references, source, partition)
