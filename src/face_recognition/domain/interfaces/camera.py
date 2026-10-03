"""Camera contract. Pixel format and logical dimensions accompany opaque data."""

from typing import Protocol

from face_recognition.domain.models.frame import Frame


class Camera(Protocol):
    def read(self) -> Frame | None:
        """Return the next frame, or None when capture fails."""
        ...

    def close(self) -> None: ...
