"""Generic rejection errors, binary/open-set rates and threshold selection result."""

from dataclasses import dataclass


class RejectionError(ValueError):
    pass


@dataclass(frozen=True)
class RejectionMetrics:
    known_acceptance_rate: float
    known_rejection_rate: float
    unknown_rejection_rate: float
    unknown_false_acceptance_rate: float
    binary_balanced_accuracy: float
    binary_macro_f1: float
    known_correct_identification_rate: float
    known_misidentification_rate: float
    open_set_exact_accuracy: float


@dataclass(frozen=True)
class ThresholdCalibration:
    threshold: float
    metrics: RejectionMetrics
    thresholds_evaluated: int
