"""Read-only, role-validated unknown calibration data; not a training dataset."""

from dataclasses import dataclass, field

import numpy as np

from face_recognition.application.unknown_rejection_split import UnknownRejectionError, validate_unknown_split
from face_recognition.domain.models.identity_split import IdentityPartition, IdentitySplitManifest, SplitSampleReference
from face_recognition.domain.models.unknown_rejection import UnknownRejectionSplit


@dataclass(frozen=True, eq=False)
class UnknownCalibrationData:
    X: np.ndarray = field(repr=False)
    samples: tuple[SplitSampleReference, ...] = field(repr=False)
    source: IdentitySplitManifest = field(repr=False)
    partition: UnknownRejectionSplit = field(repr=False)

    def __post_init__(self) -> None:
        validate_unknown_split(self.source, self.partition)
        expected = tuple(row for row in self.source.samples if row.partition is IdentityPartition.UNKNOWN_POOL
                         and row.person_id in self.partition.specification.calibration_ids)
        if (self.samples != expected or not expected or self.X.ndim != 2 or self.X.dtype != np.float32
                or self.X.shape[0] != len(expected) or not self.X.shape[1]):
            raise UnknownRejectionError("Unknown calibration rows must exactly match authorized calibration references")
        if any(not np.isfinite(row).all() for row in self.X):
            raise UnknownRejectionError("Unknown calibration features must be finite")
        self.X.setflags(write=False)

    def __repr__(self) -> str:
        return f"UnknownCalibrationData(samples={len(self.samples)}, features={self.X.shape[1]})"
