"""Reusable classifier configuration, measurements and dependency versions."""

from dataclasses import dataclass
from math import isfinite

ParameterValue = str | int | float | bool | None


@dataclass(frozen=True)
class ClassifierConfiguration:
    name: str
    classifier_parameters: tuple[tuple[str, ParameterValue], ...]
    scaler_parameters: tuple[tuple[str, ParameterValue], ...] | None


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float

    def __post_init__(self) -> None:
        if any(not isfinite(value) or not 0 <= value <= 1 for value in (
            self.accuracy, self.macro_precision, self.macro_recall, self.macro_f1,
        )):
            raise ValueError("Classification metrics must be finite and between zero and one")


@dataclass(frozen=True)
class EvaluationTiming:
    fit_seconds: float
    prediction_seconds: float
    mean_prediction_latency_ms: float

    def __post_init__(self) -> None:
        if any(not isfinite(value) or value < 0 for value in (
            self.fit_seconds, self.prediction_seconds, self.mean_prediction_latency_ms,
        )):
            raise ValueError("Evaluation timings must be finite and nonnegative")


@dataclass(frozen=True)
class FoldMeasurements:
    metrics: ClassificationMetrics
    timing: EvaluationTiming
    confusion_matrix: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class EvaluationVersions:
    python: str
    numpy: str
    scipy: str
    scikit_learn: str
