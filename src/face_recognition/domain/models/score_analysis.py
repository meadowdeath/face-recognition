"""Aggregate calibration score distributions and diagnostic ROC/EER metadata."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoreDistribution:
    count: int
    mean: float
    std: float
    median: float
    p10: float
    p25: float
    p75: float
    p90: float


@dataclass(frozen=True)
class EqualErrorPoint:
    eer: float
    threshold: float
    far: float
    frr: float


@dataclass(frozen=True)
class ScoreSeparation:
    """Threshold-independent diagnostics only; contains no operational calibration."""
    roc_auc: float
    equal_error_point: EqualErrorPoint
    known_distribution: ScoreDistribution
    unknown_distribution: ScoreDistribution
    diagnostic_only: bool = True
