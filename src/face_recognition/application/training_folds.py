"""Development-only matrices and session-aware folds. No fitting or model evaluation.

Fit selection/preprocessing/classifiers only on a fold's fit rows. Validation
may only be transformed by fit-learned parameters. Neither protected partition
belongs in these objects. Arrays are read-only API guards, not access controls.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from face_recognition.domain.models.dataset import CONDITIONS, validate_identifier
from face_recognition.domain.models.identity_split import IdentityPartition, SplitSampleReference


class TrainingFoldError(ValueError):
    pass


@dataclass(frozen=True, eq=False)
class DevelopmentDataset:
    X: np.ndarray = field(repr=False)
    y: np.ndarray = field(repr=False)
    samples: tuple[SplitSampleReference, ...] = field(repr=False)
    label_to_person: tuple[str, ...] = field(repr=False)
    training_sessions: tuple[str, ...]

    def __post_init__(self) -> None:
        if (self.X.ndim != 2 or not self.X.shape[0] or not self.X.shape[1] or self.X.dtype != np.float32
                or self.y.ndim != 1 or self.y.dtype != np.int32 or len(self.y) != len(self.samples)
                or self.X.shape[0] != len(self.samples)):
            raise TrainingFoldError("Development matrix/label shapes or dtypes are invalid")
        if (not self.label_to_person or self.label_to_person != tuple(sorted(set(self.label_to_person)))
                or not self.training_sessions or self.training_sessions != tuple(sorted(set(self.training_sessions)))):
            raise TrainingFoldError("Identity labels and training sessions must be canonical and nonempty")
        for identifier in (*self.label_to_person, *self.training_sessions):
            validate_identifier(identifier)
        labels = {person: index for index, person in enumerate(self.label_to_person)}
        observed = {person: set() for person in self.label_to_person}
        previous = None
        for index, sample in enumerate(self.samples):
            if (sample.partition is not IdentityPartition.KNOWN_TRAIN_POOL or sample.person_id not in labels
                    or sample.session_id not in self.training_sessions or self.y[index] != labels[sample.person_id]):
                raise TrainingFoldError("Protected or inconsistent sample/label in development data")
            key = (sample.session_id, sample.person_id, CONDITIONS.index(sample.condition), sample.sample_index)
            if previous is not None and key <= previous:
                raise TrainingFoldError("Development rows must follow deterministic session-first order")
            previous = key
            observed[sample.person_id].add(sample.session_id)
            if not np.isfinite(self.X[index]).all():
                raise TrainingFoldError("Development features must be finite")
        if any(sessions != set(self.training_sessions) for sessions in observed.values()):
            raise TrainingFoldError("Every development identity must have all configured training sessions")
        self.X.setflags(write=False)
        self.y.setflags(write=False)

    def __repr__(self) -> str:
        return (f"DevelopmentDataset(samples={len(self.samples)}, features={self.X.shape[1]}, "
                f"identities={len(self.label_to_person)})")


@dataclass(frozen=True, eq=False)
class TrainingRows:
    dataset: DevelopmentDataset = field(repr=False)
    indices: tuple[int, ...] = field(repr=False)

    def __post_init__(self) -> None:
        if (not self.indices or self.indices != tuple(sorted(set(self.indices)))
                or any(type(index) is not int or not 0 <= index < len(self.dataset.samples) for index in self.indices)):
            raise TrainingFoldError("Fold indices must be ordered, unique and valid")

    def _select(self, matrix: np.ndarray) -> np.ndarray:
        first, last = self.indices[0], self.indices[-1]
        if last - first + 1 == len(self.indices):
            return matrix[first:last + 1]  # Ordinary two-session folds share storage.
        selected = matrix[np.fromiter(self.indices, dtype=np.intp)]
        selected.setflags(write=False)
        return selected

    @property
    def X(self) -> np.ndarray:
        """Read-only view when contiguous; on-demand gather otherwise."""
        return self._select(self.dataset.X)

    @property
    def y(self) -> np.ndarray:
        return self._select(self.dataset.y)

    @property
    def metadata(self) -> tuple[SplitSampleReference, ...]:
        return tuple(self.dataset.samples[index] for index in self.indices)

    @property
    def identity_count(self) -> int:
        return len({self.dataset.samples[index].person_id for index in self.indices})

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return self.X, self.y

    def __repr__(self) -> str:
        return f"TrainingRows(samples={len(self.indices)}, identities={self.identity_count})"


@dataclass(frozen=True)
class SessionFoldSpec:
    name: str
    fit_sessions: tuple[str, ...]
    validation_sessions: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise TrainingFoldError("Fold name must not be empty")
        for name in ("fit_sessions", "validation_sessions"):
            values = tuple(getattr(self, name))
            if not values or len(values) != len(set(values)):
                raise TrainingFoldError("Fold session sets must be nonempty and unique")
            for value in values:
                validate_identifier(value)
            object.__setattr__(self, name, tuple(sorted(values)))
        if set(self.fit_sessions) & set(self.validation_sessions):
            raise TrainingFoldError("Fit and validation sessions must be disjoint")


@dataclass(frozen=True, eq=False)
class SessionAwareFold:
    name: str
    fit_session_ids: tuple[str, ...]
    validation_session_ids: tuple[str, ...]
    fit: TrainingRows = field(repr=False)
    validation: TrainingRows = field(repr=False)

    def __repr__(self) -> str:
        return f"SessionAwareFold(name={self.name!r}, fit_samples={len(self.fit.indices)}, validation_samples={len(self.validation.indices)})"


def create_session_folds(data: DevelopmentDataset,
                         specifications: Sequence[SessionFoldSpec] | None = None) -> tuple[SessionAwareFold, ...]:
    """Default A/B use the two configured training sessions, never fixed names.

    For other training-session counts supply explicit whole-session fold specs.
    Every fold covers the full development pool exactly once, with disjoint
    session sets. No random state or per-sample split operation is used.
    """
    if specifications is None:
        if len(data.training_sessions) != 2:
            raise TrainingFoldError("Default opposite folds require two training sessions; supply explicit fold specifications")
        first, second = data.training_sessions
        specifications = (SessionFoldSpec("Fold A", (first,), (second,)), SessionFoldSpec("Fold B", (second,), (first,)))
    if not specifications or len({spec.name for spec in specifications}) != len(specifications):
        raise TrainingFoldError("Supply uniquely named fold specifications")
    folds = []
    for spec in specifications:
        if set(spec.fit_sessions) | set(spec.validation_sessions) != set(data.training_sessions):
            raise TrainingFoldError("A fold must cover only and all configured development sessions")
        fit = tuple(index for index, sample in enumerate(data.samples) if sample.session_id in spec.fit_sessions)
        validation = tuple(index for index, sample in enumerate(data.samples) if sample.session_id in spec.validation_sessions)
        if set(fit) & set(validation) or len(fit) + len(validation) != len(data.samples):
            raise TrainingFoldError("Fold sample coverage is inconsistent")
        folds.append(SessionAwareFold(spec.name, spec.fit_sessions, spec.validation_sessions,
                                      TrainingRows(data, fit), TrainingRows(data, validation)))
    return tuple(folds)
