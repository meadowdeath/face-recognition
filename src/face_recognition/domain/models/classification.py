"""Persisted runtime version metadata; retain its historical type identity."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationVersions:
    python: str
    numpy: str
    scipy: str
    scikit_learn: str
