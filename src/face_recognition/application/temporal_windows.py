"""Boundary-safe offline windows, plurality votes and median confidences."""

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from face_recognition.application.unknown_rejection_split import UnknownRejectionError
from face_recognition.domain.models.dataset import CONDITIONS
from face_recognition.domain.models.identity_split import SplitSampleReference
from face_recognition.domain.models.temporal_open_set import WINDOW_SIZES, VoteRates, VoteStability, WindowCounts


@dataclass(frozen=True)
class SourceWindow:
    person_id: str = field(repr=False)
    session_id: str = field(repr=False)
    condition: str
    indices: tuple[int, ...] = field(repr=False)


@dataclass(frozen=True)
class WindowPlan:
    size: int
    counts: WindowCounts
    condition_counts: tuple[tuple[str, WindowCounts], ...]
    windows: tuple[SourceWindow, ...] = field(repr=False)


def build_windows(samples: tuple[SplitSampleReference, ...], size: int) -> WindowPlan:
    if type(size) is not int or size not in WINDOW_SIZES:
        raise UnknownRejectionError("Temporal window size must be one of 1,3,5,7")
    groups = defaultdict(list)
    for position, sample in enumerate(samples):
        if sample.condition not in CONDITIONS or type(sample.sample_index) is not int or sample.sample_index < 0:
            raise UnknownRejectionError("Invalid temporal source metadata")
        groups[sample.person_id, sample.session_id, sample.condition].append((sample.sample_index, position))
    windows, condition_counts = [], defaultdict(lambda: [0, 0, 0, 0])
    for person, session, condition in sorted(groups, key=lambda key: (key[0], key[1], CONDITIONS.index(key[2]))):
        rows = sorted(groups[person, session, condition])
        indices = [index for index, _ in rows]
        if indices != list(range(indices[0], indices[0] + len(indices))):
            raise UnknownRejectionError("Temporal groups require unique contiguous sample indices")
        used = len(rows) // size * size
        for start in range(0, used, size):
            windows.append(SourceWindow(person, session, condition, tuple(position for _, position in rows[start:start + size])))
        counter = condition_counts[condition]
        for index, value in enumerate((len(rows), used, len(rows) - used, used // size)):
            counter[index] += value
    used = sum(len(window.indices) for window in windows)
    return WindowPlan(size, WindowCounts(len(samples), used, len(samples) - used, len(windows)),
                       tuple((condition, WindowCounts(*condition_counts[condition]))
                             for condition in CONDITIONS if condition in condition_counts), tuple(windows))


def vote_identity(predicted: np.ndarray, decisions: np.ndarray) -> tuple[int, str]:
    predicted, decisions = np.asarray(predicted), np.asarray(decisions)
    if (predicted.ndim != 1 or not len(predicted) or decisions.ndim != 2 or decisions.shape[0] != len(predicted)
            or decisions.shape[1] < 2 or not np.isfinite(decisions).all()
            or not np.issubdtype(predicted.dtype, np.integer) or np.any(predicted < 0) or np.any(predicted >= decisions.shape[1])):
        raise UnknownRejectionError("Invalid temporal predictions or decision scores")
    counts = np.bincount(predicted, minlength=decisions.shape[1])
    tied = np.flatnonzero(counts == counts.max())
    if len(tied) == 1:
        return int(tied[0]), "unanimous" if counts.max() == len(predicted) else "majority"
    with np.errstate(over="ignore"):
        sums = decisions[:, tied].sum(axis=0, dtype=np.float64)
    if not np.isfinite(sums).all():
        raise UnknownRejectionError("Nonfinite summed temporal decision scores")
    return int(tied[np.argmax(sums)]), "tie"


@dataclass(frozen=True, eq=False)
class WindowObservations:
    predicted: np.ndarray = field(repr=False)
    confidence: tuple[tuple[str, np.ndarray], ...] = field(repr=False)
    conditions: tuple[str, ...] = field(repr=False)
    vote_stability: VoteStability


def aggregate_windows(plan: WindowPlan, predicted: np.ndarray, decisions: np.ndarray,
                       confidence: tuple[tuple[str, np.ndarray], ...]) -> WindowObservations:
    if (predicted.ndim != 1 or len(predicted) != plan.counts.source_samples
            or decisions.ndim != 2 or len(decisions) != len(predicted) or decisions.shape[1] < 2
            or not np.issubdtype(predicted.dtype, np.integer) or np.any(predicted < 0) or np.any(predicted >= decisions.shape[1])
            or not np.isfinite(decisions).all() or not confidence
            or len({name for name, _ in confidence}) != len(confidence)
            or any(np.asarray(values).shape != (len(predicted),) or not np.isfinite(values).all() for _, values in confidence)):
        raise UnknownRejectionError("Temporal inference arrays must match source metadata and be finite")
    winners = np.empty(len(plan.windows), dtype=np.int32)
    medians = [(name, np.empty(len(plan.windows), dtype=np.float64)) for name, _ in confidence]
    categories = {"unanimous": 0, "majority": 0, "tie": 0}
    for position, window in enumerate(plan.windows):
        indices = np.asarray(window.indices, dtype=np.intp)
        winner, category = vote_identity(predicted[indices], decisions[indices])
        winners[position] = winner
        categories[category] += 1
        for (_, values), (_, output) in zip(confidence, medians):
            output[position] = np.median(values[indices])
    n = len(plan.windows)
    rates = VoteRates(*(categories[key] / n if n else 0.0 for key in ("unanimous", "majority", "tie")))
    stability = VoteStability(categories["unanimous"], categories["majority"], categories["tie"], rates)
    return WindowObservations(winners, tuple(medians), tuple(window.condition for window in plan.windows), stability)
