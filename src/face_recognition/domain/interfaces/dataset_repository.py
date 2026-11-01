from collections.abc import Sequence
from typing import Protocol

from face_recognition.domain.models.dataset import RawLandmarkSample, SessionConfig, SessionManifest


class DatasetRepository(Protocol):
    def open_session(self, config: SessionConfig) -> SessionManifest:
        """Create or validate/resume a session without overwriting completed data."""
        ...

    def save_condition(
        self, manifest: SessionManifest, condition: str, samples: Sequence[RawLandmarkSample],
    ) -> SessionManifest:
        """Publish a complete condition, then atomically update its manifest."""
        ...
