"""Explicit feature-transform specifications and fitted structural metadata."""

from dataclasses import dataclass
from math import isfinite

from face_recognition.domain.models.classification import FoldMeasurements


@dataclass(frozen=True)
class ReductionConfiguration:
    method: str
    k: int | None = None
    explained_variance: float | None = None

    def __post_init__(self) -> None:
        if self.method == "baseline":
            valid = self.k is None and self.explained_variance is None
        elif self.method == "anova":
            valid = type(self.k) is int and self.k > 0 and self.explained_variance is None
        elif self.method == "pca":
            valid = (self.k is None and type(self.explained_variance) in (int, float)
                     and isfinite(self.explained_variance) and 0 < self.explained_variance < 1)
        else:
            valid = False
        if not valid:
            raise ValueError("Invalid feature reduction configuration")

    @property
    def identifier(self) -> str:
        if self.method == "anova":
            return f"anova_k_{self.k}"
        if self.method == "pca":
            return f"pca_{self.explained_variance}"
        return "baseline"


@dataclass(frozen=True)
class FittedReductionMetadata:
    retained_features: int
    selected_feature_indices: tuple[int, ...] | None = None
    actual_explained_variance: float | None = None


@dataclass(frozen=True)
class ReducedMeasurements:
    measurements: FoldMeasurements
    transformation: FittedReductionMetadata
