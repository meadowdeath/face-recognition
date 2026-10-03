from dataclasses import dataclass


@dataclass(frozen=True)
class Prediction:
    label: str | None
    score: float | None = None
