"""Serialized fitted container: historical module path required by deployed bundles."""

from dataclasses import dataclass, field


@dataclass(frozen=True, eq=False)
class FittedV3Identity:
    pipeline: object = field(repr=False)
    centroid: object = field(repr=False)
    mahalanobis: object = field(repr=False)
    identity_count: int
    methods: tuple[str, ...]
