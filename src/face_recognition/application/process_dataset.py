"""Offline normalization and baseline orchestration; no JSON, paths or vendor APIs."""

from collections.abc import Callable
from dataclasses import replace

from face_recognition.application.normalization_experiment import LandmarkNormalization
from face_recognition.domain.interfaces.dataset_processing import ProcessedDatasetRepository, RawDatasetSource, SessionT
from face_recognition.domain.interfaces.feature_extractor import FeatureExtractor
from face_recognition.domain.models.dataset import RawLandmarkSample, validate_raw_landmarks
from face_recognition.domain.models.processed_sample import (
    BaselineFeatureSample, NormalizedLandmarkSample, ProcessingIssue, ProcessingSummary, SampleProvenance,
)


class DatasetProcessingError(ValueError):
    """Safe category/context only; underlying biometric-bearing exceptions are suppressed."""

    def __init__(self, issue: ProcessingIssue, summary: ProcessingSummary | None = None) -> None:
        super().__init__(issue.describe())
        self.issue = issue
        self.summary = summary


class DatasetProcessor:
    def __init__(self, normalizer_factory: Callable[[int, int], LandmarkNormalization],
                 feature_extractor: FeatureExtractor) -> None:
        self.normalizer_factory = normalizer_factory
        self.feature_extractor = feature_extractor

    def process_sample(self, sample: RawLandmarkSample) -> tuple[NormalizedLandmarkSample, BaselineFeatureSample]:
        context = dict(person_id=sample.person_id, session_id=sample.session_id,
                       condition=sample.condition, sample_index=sample.sample_index)
        try:
            provenance = SampleProvenance.from_raw(sample)
            validate_raw_landmarks(sample.landmarks)
        except Exception:
            raise DatasetProcessingError(ProcessingIssue("input_validation_failed", **context)) from None
        try:
            normalizer = self.normalizer_factory(sample.inference_width, sample.inference_height)
            landmarks = normalizer.normalize(sample.landmarks)
            if len(landmarks.points) != len(sample.landmarks.points):
                raise ValueError
            normalized = NormalizedLandmarkSample(provenance, landmarks)
        except Exception:
            raise DatasetProcessingError(ProcessingIssue("normalization_failed", **context)) from None
        try:
            baseline = BaselineFeatureSample(provenance, self.feature_extractor.extract(landmarks), len(landmarks.points))
        except Exception:
            raise DatasetProcessingError(ProcessingIssue("feature_extraction_failed", **context)) from None
        return normalized, baseline

    def process(self, source: RawDatasetSource[SessionT], repository: ProcessedDatasetRepository) -> ProcessingSummary:
        """Fail fast; no partial run is published as a completed output.

        Counts describe successfully written sample pairs. Source-level failures
        may have no trustworthy sample index because the reader validates whole
        conditions before returning. Decoding errors carry their expected file
        index when available; file-level errors retain session/condition context.
        """
        participants: set[str] = set()
        sessions: set[tuple[str, str]] = set()
        conditions: set[tuple[str, str, str]] = set()
        total = 0
        context = ProcessingIssue("source_discovery_failed")

        def summary() -> ProcessingSummary:
            return ProcessingSummary(len(participants), len(sessions), len(conditions), total)

        try:
            repository.begin(source.root)
        except Exception:
            raise DatasetProcessingError(ProcessingIssue("output_initialization_failed"), summary()) from None
        try:
            discovered = source.discover_sessions()
            for session in discovered:
                config = session.manifest.config
                for condition in session.manifest.completed_conditions:
                    context = ProcessingIssue("source_validation_failed", config.person_id, config.session_id, condition)
                    samples = source.read_condition(session, condition)
                    for sample in samples:
                        normalized, baseline = self.process_sample(sample)
                        context = ProcessingIssue("output_write_failed", sample.person_id, sample.session_id,
                                                  sample.condition, sample.sample_index)
                        repository.write_sample(normalized, baseline)
                        participants.add(sample.person_id)
                        sessions.add((sample.person_id, sample.session_id))
                        conditions.add((sample.person_id, sample.session_id, sample.condition))
                        total += 1
            context = ProcessingIssue("output_completion_failed")
            repository.complete(summary())
            return summary()
        except Exception as exc:
            issue = exc.issue if isinstance(exc, DatasetProcessingError) else context
            if issue.category == "source_validation_failed":
                index = getattr(exc, "sample_index", None)
                if type(index) is int and index >= 0:
                    issue = replace(issue, sample_index=index)
            rejected = int(issue.category in ("input_validation_failed", "normalization_failed", "feature_extraction_failed"))
            if issue.category == "source_validation_failed" and issue.sample_index is not None:
                rejected = 1
            failed = replace(summary(), rejected_samples=rejected, error_count=1)
            try:
                repository.fail(failed, issue)
            except Exception:
                pass  # Preserve the original safe failure; staging never becomes a final run.
            raise DatasetProcessingError(issue, failed) from None
        except BaseException:
            try:
                repository.fail(replace(summary(), error_count=1), ProcessingIssue("processing_interrupted"))
            except Exception:
                pass
            raise
