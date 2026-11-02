"""Window accounting and aggregate identity-vote stability."""

from dataclasses import dataclass

WINDOW_SIZES = (1, 3, 5, 7)


@dataclass(frozen=True)
class WindowCounts:
    source_samples: int
    used_samples: int
    discarded_samples: int
    windows: int


@dataclass(frozen=True)
class VoteRates:
    unanimous_rate: float
    majority_not_unanimous_rate: float
    tie_rate: float


@dataclass(frozen=True)
class VoteStability:
    unanimous_windows: int
    majority_not_unanimous_windows: int
    tied_windows: int
    rates: VoteRates
