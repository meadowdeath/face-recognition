"""Normalized landmarks returned by a detector, without vendor types."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class FaceLandmarks:
    points: tuple[Landmark, ...]

    def __post_init__(self) -> None:
        if not self.points:
            raise ValueError("A face must contain at least one landmark")
