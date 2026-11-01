"""Boundaries used by offline processing; no camera APIs or filesystem operations."""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, TypeVar

from face_recognition.domain.models.dataset import RawLandmarkSample, SessionManifest
from face_recognition.domain.models.processed_sample import (
    BaselineFeatureSample, NormalizedLandmarkSample, ProcessingIssue, ProcessingSummary,
)


class DatasetSession(Protocol):
    @property
    def manifest(self) -> SessionManifest: ...


SessionT = TypeVar("SessionT", bound=DatasetSession)


class RawDatasetSource(Protocol[SessionT]):
    @property
    def root(self) -> Path: ...

    def discover_sessions(self) -> Sequence[SessionT]: ...

    def read_condition(self, session: SessionT, condition: str) -> Sequence[RawLandmarkSample]: ...


class ProcessedDatasetRepository(Protocol):
    def begin(self, source_root: Path) -> None: ...

    def write_sample(self, normalized: NormalizedLandmarkSample, baseline: BaselineFeatureSample) -> None: ...

    def complete(self, summary: ProcessingSummary) -> None: ...

    def fail(self, summary: ProcessingSummary, issue: ProcessingIssue) -> None: ...
