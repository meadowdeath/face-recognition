"""Final-only populations. Never usable as development/training data."""

from dataclasses import dataclass, field

import numpy as np

from face_recognition.application.unknown_rejection_split import validate_unknown_split
from face_recognition.domain.models.identity_split import IdentityPartition, IdentitySplitManifest, SplitSampleReference
from face_recognition.domain.models.unknown_rejection import UnknownRejectionSplit


class FinalEvaluationError(ValueError):
    pass


@dataclass(frozen=True, eq=False)
class FinalHoldoutData:
    known_X: np.ndarray = field(repr=False)
    known_y: np.ndarray = field(repr=False)
    known_samples: tuple[SplitSampleReference, ...] = field(repr=False)
    unknown_X: np.ndarray = field(repr=False)
    unknown_samples: tuple[SplitSampleReference, ...] = field(repr=False)
    class_ids: tuple[str, ...] = field(repr=False)
    source: IdentitySplitManifest = field(repr=False)
    partition: UnknownRejectionSplit = field(repr=False)

    def __post_init__(self):
        validate_unknown_split(self.source, self.partition)
        known = tuple(s for s in self.source.samples if s.partition is IdentityPartition.KNOWN_TEST)
        unknown = tuple(s for s in self.source.samples if s.partition is IdentityPartition.UNKNOWN_POOL
                        and s.person_id in self.partition.specification.holdout_ids)
        if (not known or not unknown or self.known_samples != known or self.unknown_samples != unknown
                or self.class_ids != self.source.specification.known_ids
                or self.known_y.dtype != np.int32 or self.known_y.shape != (len(known),)):
            raise FinalEvaluationError("Final populations must match the held-out manifest references exactly")
        labels = {person: index for index, person in enumerate(self.class_ids)}
        if not np.array_equal(self.known_y, np.fromiter((labels[s.person_id] for s in known), dtype=np.int32)):
            raise FinalEvaluationError("Final known labels do not match the frozen model mapping")
        for X, samples in ((self.known_X, known), (self.unknown_X, unknown)):
            if X.ndim != 2 or X.dtype != np.float32 or X.shape[0] != len(samples) or not X.shape[1] or not np.isfinite(X).all():
                raise FinalEvaluationError("Invalid final holdout matrix")
            X.setflags(write=False)
        if self.known_X.shape[1] != self.unknown_X.shape[1]:
            raise FinalEvaluationError("Final population feature dimensions disagree")
        self.known_y.setflags(write=False)
