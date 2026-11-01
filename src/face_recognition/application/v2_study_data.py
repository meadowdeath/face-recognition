"""Explicit V2 role promotion; no weakening of V1 development/holdout readers."""

from dataclasses import dataclass, field

import numpy as np

from face_recognition.application.unknown_rejection_split import validate_unknown_split
from face_recognition.domain.models.dataset import CONDITIONS
from face_recognition.domain.models.identity_split import IdentityPartition
from face_recognition.domain.models.v2_diagnostics import StudyFold


class V2StudyError(ValueError):
    pass


def known_references(source):
    return tuple(sorted((s for s in source.samples if s.person_id in source.specification.known_ids),
                        key=lambda s: (s.session_id, s.person_id, CONDITIONS.index(s.condition), s.sample_index)))


def study_folds(source) -> tuple[StudyFold, ...]:
    sessions = source.specification.train_sessions + source.specification.known_test_sessions
    if (len(source.specification.train_sessions) != 2 or len(source.specification.known_test_sessions) != 1
            or len(sessions) != 3 or len(set(sessions)) != 3):
        raise V2StudyError("V2 study requires exactly the three original known sessions; future sessions are excluded")
    return tuple(StudyFold(f"Fold {name}", tuple(s for s in sessions if s != held), held)
                 for name, held in zip("ABC", reversed(sessions)))


@dataclass(frozen=True, eq=False)
class V2StudyData:
    known_X: np.ndarray = field(repr=False)
    known_y: np.ndarray = field(repr=False)
    known_samples: tuple = field(repr=False)
    unknown_X: np.ndarray = field(repr=False)
    unknown_samples: tuple = field(repr=False)
    source: object = field(repr=False)
    partition: object = field(repr=False)

    def __post_init__(self):
        validate_unknown_split(self.source, self.partition)
        folds = study_folds(self.source)
        expected_unknown = tuple(s for s in self.source.samples if s.partition is IdentityPartition.UNKNOWN_POOL)
        if self.known_samples != known_references(self.source) or self.unknown_samples != expected_unknown:
            raise V2StudyError("V2 rows must exactly cover original known sessions and all original unknown identities")
        ids = self.source.specification.known_ids
        if len(ids) < 2 or len(self.source.specification.unknown_ids) < 2:
            raise V2StudyError("At least two known and two unknown identities are required")
        labels = {p: i for i, p in enumerate(ids)}
        if self.known_y.dtype != np.int32 or not np.array_equal(self.known_y, [labels[s.person_id] for s in self.known_samples]):
            raise V2StudyError("V2 label mapping is inconsistent")
        for X, rows in ((self.known_X, self.known_samples), (self.unknown_X, self.unknown_samples)):
            if X.ndim != 2 or X.dtype != np.float32 or X.shape != (len(rows), 1434) or not np.isfinite(X).all():
                raise V2StudyError("V2 requires finite all_xyz_v1 matrices with 1434 features")
            X.setflags(write=False)
        for person in ids:
            if {s.session_id for s in self.known_samples if s.person_id == person} != {f.calibration_session for f in folds}:
                raise V2StudyError("Every known identity must have all three original sessions")
        self.known_y.setflags(write=False)
