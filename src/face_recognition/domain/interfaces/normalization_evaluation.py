from collections.abc import Sequence
from typing import Protocol

from face_recognition.domain.models.processed_sample import NormalizedLandmarkSample

ConditionKey = tuple[str, str, str]


class NormalizedDatasetSource(Protocol):
    @property
    def expected_sample_count(self) -> int: ...

    def discover_conditions(self) -> Sequence[ConditionKey]: ...

    def read_condition(self, key: ConditionKey) -> Sequence[NormalizedLandmarkSample]: ...
