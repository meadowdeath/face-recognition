from collections.abc import Iterator
from typing import Protocol

from face_recognition.domain.models.identity_split import FeatureSampleMetadata


class FeatureMetadataSource(Protocol):
    @property
    def representation(self) -> str: ...

    @property
    def expected_sample_count(self) -> int: ...

    def iter_metadata(self) -> Iterator[FeatureSampleMetadata]: ...
