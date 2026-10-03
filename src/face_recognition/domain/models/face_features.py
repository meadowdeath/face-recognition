from dataclasses import dataclass


@dataclass(frozen=True)
class FaceFeatures:
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("A feature vector cannot be empty")
