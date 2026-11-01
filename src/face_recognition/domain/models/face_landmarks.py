"""Vendor-independent landmarks; their coordinate space depends on the producer.

Detector output uses MediaPipe's input-image-normalized coordinates: x is
width-normalized, y is height-normalized, and z uses approximately x's scale.
This does not normalize facial translation, scale, or roll for identity
features. The landmark normalizer uses explicit image dimensions to correct
aspect ratio and returns a separate eye-centered, interocular-scaled and
roll-corrected instance of the same model.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class FaceLandmarks:
    """Immutable ordered points in a producer-defined coordinate space.

    Neither image normalization nor identity normalization is applied here.
    """

    points: tuple[Landmark, ...]

    def __post_init__(self) -> None:
        if not self.points:
            raise ValueError("A face must contain at least one landmark")
