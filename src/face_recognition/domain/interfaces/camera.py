"""Camera contract. Frames are BGR uint8 arrays at adapter boundaries."""

from typing import Any, Protocol

Frame = Any  # Kept opaque so domain never imports NumPy or OpenCV.


class Camera(Protocol):
    def read(self) -> Frame | None:
        """Return the next frame, or None when capture fails."""
        ...

    def close(self) -> None: ...
