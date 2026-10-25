"""Separate derived JSONL representations, atomically published as one fresh run."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
from typing import TextIO

from face_recognition.config.settings import PROJECT_ROOT
from face_recognition.domain.models.processed_sample import (
    BASELINE_REPRESENTATION, NORMALIZED_REPRESENTATION, PROCESSED_SCHEMA_VERSION, REPRESENTATIONS,
    BaselineFeatureSample, NormalizedLandmarkSample, ProcessingIssue, ProcessingSummary, SampleProvenance,
)


class ProcessedDatasetPersistenceError(ValueError):
    """Output configuration/state is unsafe or inconsistent."""


def _record_metadata(provenance: SampleProvenance, representation: str, landmark_count: int) -> dict:
    return {"schema_version": PROCESSED_SCHEMA_VERSION, "representation": representation,
            **asdict(provenance), "source_relative_path": provenance.source_relative_path,
            "landmark_count": landmark_count}


def normalized_to_json(sample: NormalizedLandmarkSample) -> str:
    return json.dumps({
        **_record_metadata(sample.provenance, NORMALIZED_REPRESENTATION, len(sample.landmarks.points)),
        "normalized_landmarks": [[point.x, point.y, point.z] for point in sample.landmarks.points],
    }, allow_nan=False, separators=(",", ":"))


def baseline_to_json(sample: BaselineFeatureSample) -> str:
    return json.dumps({
        **_record_metadata(sample.provenance, BASELINE_REPRESENTATION, sample.landmark_count),
        "feature_count": len(sample.features.values), "features": sample.features.values,
    }, allow_nan=False, separators=(",", ":"))


class FilesystemProcessedDatasetRepository:
    """Write only derived data. Source/output roots must be disjoint.

    output_root must not already exist, even if empty. A sibling .tmp directory
    holds both representations until the entire run succeeds. Failed staging
    directories are retained with failed metadata where possible, never treated
    as final output. There is no append, overwrite, resume or automatic deletion.
    Source files and directory timestamps are never intentionally changed.
    """

    def __init__(self, output_root: Path, source_root: Path) -> None:
        if Path(output_root).is_symlink():
            raise ProcessedDatasetPersistenceError("Output root must not be a symlink")
        self.output_root = Path(output_root).resolve()
        self.source_root = Path(source_root).resolve()
        self.staging_root: Path | None = None
        self._started = False
        self._published = False
        self._failed = False
        self._streams: tuple[TextIO, ...] = ()
        self._last_provenance: SampleProvenance | None = None
        self._participants: set[str] = set()
        self._sessions: set[tuple[str, str]] = set()
        self._conditions: set[tuple[str, str, str]] = set()
        self._total = 0
        self._validate_paths()

    def _validate_paths(self) -> None:
        source, output = self.source_root.resolve(), self.output_root.resolve()
        if not source.is_dir():
            raise ProcessedDatasetPersistenceError("Source root must be an existing raw dataset directory")
        if source.is_relative_to(output) or output.is_relative_to(source):
            raise ProcessedDatasetPersistenceError("Source and output roots must not overlap")
        if output.is_relative_to(PROJECT_ROOT.resolve()):
            raise ProcessedDatasetPersistenceError("Derived output must remain outside the Git repository")
        if self.output_root.exists() or self.output_root.is_symlink():
            raise ProcessedDatasetPersistenceError("Output root already exists; choose a new output root")

    @staticmethod
    def _sync_directory(directory: Path) -> None:
        if os.name == "posix":
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

    def begin(self, source_root: Path | None = None) -> None:
        if self._started:
            raise ProcessedDatasetPersistenceError("A repository instance supports one processing run")
        if source_root is not None and Path(source_root).resolve() != self.source_root.resolve():
            raise ProcessedDatasetPersistenceError("Configured source root does not match the reader root")
        self._validate_paths()
        self.output_root.parent.mkdir(parents=True, exist_ok=True)
        self.staging_root = Path(tempfile.mkdtemp(prefix=self.output_root.name + ".", suffix=".tmp",
                                                dir=self.output_root.parent))
        self._started = True
        for representation in REPRESENTATIONS:
            (self.staging_root / representation).mkdir()
        self._write_manifests("in_progress", ProcessingSummary())

    def _require_active(self) -> Path:
        if not self._started or self._published or self._failed or self.staging_root is None:
            raise ProcessedDatasetPersistenceError("Processing repository is not active")
        return self.staging_root

    def _close_streams(self) -> None:
        streams, self._streams = self._streams, ()
        failure = None
        for stream in streams:
            try:
                stream.flush()
                os.fsync(stream.fileno())
            except OSError as exc:
                failure = exc
            finally:
                try:
                    stream.close()
                except OSError as exc:
                    failure = exc
        if failure is not None:
            raise failure

    def write_sample(self, normalized: NormalizedLandmarkSample, baseline: BaselineFeatureSample) -> None:
        stage = self._require_active()
        provenance = normalized.provenance
        if provenance != baseline.provenance or len(normalized.landmarks.points) != baseline.landmark_count:
            raise ProcessedDatasetPersistenceError("Derived sample provenance/counts must match")
        previous = self._last_provenance
        same_condition = previous is not None and previous.condition_key == provenance.condition_key
        expected_index = previous.sample_index + 1 if same_condition else 0
        if provenance.sample_index != expected_index or (previous is not None and provenance.condition_key < previous.condition_key):
            raise ProcessedDatasetPersistenceError("Samples must follow deterministic session/condition/index order")
        # Encode/validate both representations before writing either record.
        lines = (normalized_to_json(normalized), baseline_to_json(baseline))
        if not same_condition:
            self._close_streams()
            opened: list[TextIO] = []
            try:
                for representation in REPRESENTATIONS:
                    path = stage / representation / provenance.source_relative_path
                    path.parent.mkdir(parents=True, exist_ok=True)
                    opened.append(path.open("x", encoding="utf-8", newline="\n"))
            except BaseException:
                for stream in opened:
                    stream.close()
                raise
            self._streams = tuple(opened)
        for stream, line in zip(self._streams, lines):
            stream.write(line + "\n")
        self._last_provenance = provenance
        self._participants.add(provenance.person_id)
        self._sessions.add((provenance.person_id, provenance.session_id))
        self._conditions.add((provenance.person_id, provenance.session_id, provenance.condition))
        self._total += 1

    def _write_json(self, path: Path, data: dict) -> None:
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(data, allow_nan=False, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    def _write_manifests(self, state: str, summary: ProcessingSummary, issue: ProcessingIssue | None = None) -> None:
        if self.staging_root is None:
            raise ProcessedDatasetPersistenceError("No staging directory")
        common = {"schema_version": PROCESSED_SCHEMA_VERSION, "completion_state": state,
                  **asdict(summary), "normalization": "LandmarkNormalizer",
                  "feature_extraction": "LandmarkFeatureExtractor",
                  "inference_dimensions": "per_source_sample"}
        if issue is not None:
            common["failure"] = asdict(issue)
        self._write_json(self.staging_root / "processing_manifest.json",
                         {**common, "representations": list(REPRESENTATIONS),
                          "condition_files_per_representation": summary.conditions,
                          "total_condition_files": summary.conditions * len(REPRESENTATIONS)})
        for representation in REPRESENTATIONS:
            self._write_json(self.staging_root / representation / "processing_manifest.json",
                             {**common, "representation": representation, "condition_files": summary.conditions})

    def complete(self, summary: ProcessingSummary) -> None:
        stage = self._require_active()
        observed = ProcessingSummary(len(self._participants), len(self._sessions), len(self._conditions), self._total)
        if summary != observed:
            raise ProcessedDatasetPersistenceError("Completion counts do not match written sample pairs")
        self._close_streams()
        self._write_manifests("completed", summary)
        # Flush staged directory entries too, before publication on Linux.
        for directory in (path for path in stage.rglob("*") if path.is_dir()):
            self._sync_directory(directory)
        self._sync_directory(stage)
        self._validate_paths()
        os.rename(stage, self.output_root)
        self._published = True
        self.staging_root = None
        self._sync_directory(self.output_root.parent)

    def fail(self, summary: ProcessingSummary, issue: ProcessingIssue) -> None:
        if not self._started or self._published or self.staging_root is None:
            return
        try:
            self._close_streams()
        finally:
            self._failed = True
            self._write_manifests("failed", summary, issue)
