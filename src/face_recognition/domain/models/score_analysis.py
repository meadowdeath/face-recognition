"""Aggregate score distributions, separation diagnostics and operating points."""

from dataclasses import dataclass

from face_recognition.domain.models.rejection import RejectionMetrics, ThresholdCalibration

CENTROID_SCALE_EPSILON = 1e-8
FAR_TARGETS = (.20, .10, .05)


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
class FarOperatingPoint:
    target_far: float
    status: str
    threshold: float | None
    metrics: RejectionMetrics | None
    reason: str | None = None


@dataclass(frozen=True)
class ScoreAnalysis:
    roc_auc: float
    equal_error_point: EqualErrorPoint
    calibration: ThresholdCalibration
    far_operating_points: tuple[FarOperatingPoint, ...]
    known_distribution: ScoreDistribution
    unknown_distribution: ScoreDistribution


@dataclass(frozen=True)
class ScoreSeparation:
    """Threshold-independent diagnostics only; contains no operational calibration."""
    roc_auc: float
    equal_error_point: EqualErrorPoint
    known_distribution: ScoreDistribution
    unknown_distribution: ScoreDistribution
    diagnostic_only: bool = True
