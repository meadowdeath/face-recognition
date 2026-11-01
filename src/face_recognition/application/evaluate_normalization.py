"""Session-local center comparisons and aggregate-only normalization quality."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from math import isfinite

from face_recognition.application.normalization_experiment import landmark_rmse, raw_comparison_geometry
from face_recognition.application.normalization_validation import RMSEStatistics, mean_landmark_template, summarize_rmse
from face_recognition.domain.interfaces.dataset_processing import RawDatasetSource, SessionT
from face_recognition.domain.interfaces.normalization_evaluation import ConditionKey, NormalizedDatasetSource
from face_recognition.domain.models.dataset import CONDITIONS, RawLandmarkSample
from face_recognition.domain.models.face_landmarks import FaceLandmarks
from face_recognition.domain.models.processed_sample import NORMALIZED_REPRESENTATION, NormalizedLandmarkSample, ProcessingIssue, SampleProvenance

TARGETED_CONDITIONS = ("near", "far", "translation_left", "translation_right", "translation_up",
                       "translation_down", "roll_left", "roll_right")
POSE_CONDITIONS = ("yaw_left", "yaw_right", "pitch_up", "pitch_down")


class NormalizationEvaluationError(ValueError):
    def __init__(self, issue: ProcessingIssue) -> None:
        super().__init__(issue.describe())
        self.issue = issue


@dataclass(frozen=True)
class ConditionQuality:
    condition: str
    sample_pairs: int
    raw: RMSEStatistics  # Width-relative units.
    comparable_raw: RMSEStatistics  # Divide raw error by this session's center eye distance.
    normalized: RMSEStatistics  # Already interocular-scaled; never normalized again.
    absolute_reduction: float
    percentage_reduction: float | None


@dataclass(frozen=True)
class NormalizationQualityReport:
    participants: int
    sessions: int
    condition_files: int
    evaluated_sample_pairs: int
    conditions: tuple[ConditionQuality, ...]
    targeted_invariance: ConditionQuality | None
    pose: ConditionQuality | None
    schema_version: int = 1
    completion_state: str = "completed"
    normalized_representation: str = NORMALIZED_REPRESENTATION
    reference: str = "Separate means of corrected raw and stored normalized center samples within each person/session"
    raw_units: str = "image_width"
    comparison_units: str = "session_center_interocular_distance"
    reduction_definition: str = "mean(raw_rmse / center_interocular_distance) - mean(normalized_rmse)"
    percentage_definition: str = "100 * reduction / mean(comparable_raw); null when denominator is zero or result nonfinite"
    standard_deviation: str = "population"
    weighting: str = "pooled sample pairs; no averaging of face coordinates across sessions or people"
    center_baseline: str = "In-sample deviations from the same center mean; descriptive, not an independent test"


@dataclass
class _Errors:
    raw: list[float] = field(default_factory=list)
    comparable: list[float] = field(default_factory=list)
    normalized: list[float] = field(default_factory=list)

    def add(self, raw: float, normalized: float, reference_distance: float) -> None:
        comparable = raw / reference_distance
        if not all(isfinite(value) for value in (raw, comparable, normalized)):
            raise ValueError("Metric must be finite")
        self.raw.append(raw)
        self.comparable.append(comparable)
        self.normalized.append(normalized)

    def summarize(self, label: str) -> ConditionQuality | None:
        if not self.raw:
            return None
        raw, comparable, normalized = (summarize_rmse(values) for values in (self.raw, self.comparable, self.normalized))
        reduction = comparable.mean - normalized.mean
        percent = 100 * (reduction / comparable.mean) if comparable.mean > 0 else None
        if percent is not None and not isfinite(percent):
            percent = None
        return ConditionQuality(label, len(self.raw), raw, comparable, normalized, reduction, percent)


def _paired(
    raw: Sequence[RawLandmarkSample], normalized: Sequence[NormalizedLandmarkSample], key: ConditionKey,
) -> tuple[tuple[RawLandmarkSample, NormalizedLandmarkSample], ...]:
    if not raw or len(raw) != len(normalized):
        raise NormalizationEvaluationError(ProcessingIssue("sample_pair_count_mismatch", *key))
    for index, (original, derived) in enumerate(zip(raw, normalized)):
        expected = SampleProvenance.from_raw(original)
        if (expected != derived.provenance or (expected.person_id, expected.session_id, expected.condition) != key
                or expected.sample_index != index or len(original.landmarks.points) != len(derived.landmarks.points)):
            raise NormalizationEvaluationError(ProcessingIssue("sample_metadata_mismatch", *key, index))
    return tuple(zip(raw, normalized))


class NormalizationQualityEvaluator:
    def __init__(self, reference_distance: Callable[[FaceLandmarks], float]) -> None:
        self.reference_distance = reference_distance

    def evaluate(self, raw: RawDatasetSource[SessionT], normalized: NormalizedDatasetSource) -> NormalizationQualityReport:
        context = ProcessingIssue("source_discovery_failed")
        try:
            sessions = raw.discover_sessions()
            if not sessions:
                raise NormalizationEvaluationError(ProcessingIssue("no_sessions_available"))
            keys = []
            for session in sessions:
                config = session.manifest.config
                if "center" not in session.manifest.completed_conditions:
                    raise NormalizationEvaluationError(ProcessingIssue("missing_center_condition", config.person_id, config.session_id))
                keys.extend((config.person_id, config.session_id, condition) for condition in session.manifest.completed_conditions)
            if set(keys) != set(normalized.discover_conditions()):
                raise NormalizationEvaluationError(ProcessingIssue("raw_normalized_catalog_mismatch"))
            buckets = {condition: _Errors() for condition in CONDITIONS}
            targeted, pose = _Errors(), _Errors()
            total = 0
            for session in sessions:
                config = session.manifest.config
                center_key = (config.person_id, config.session_id, "center")
                context = ProcessingIssue("invalid_center_reference", *center_key)
                centers = _paired(raw.read_condition(session, "center"), normalized.read_condition(center_key), center_key)
                raw_center = mean_landmark_template([
                    raw_comparison_geometry(sample.landmarks, sample.inference_width, sample.inference_height)
                    for sample, _ in centers
                ])
                normalized_center = mean_landmark_template([sample.landmarks for _, sample in centers])
                distance = self.reference_distance(raw_center)
                if not isfinite(distance) or distance <= 0:
                    raise ValueError
                for condition in session.manifest.completed_conditions:
                    key = (config.person_id, config.session_id, condition)
                    context = ProcessingIssue("condition_read_failed", *key)
                    pairs = centers if condition == "center" else _paired(
                        raw.read_condition(session, condition), normalized.read_condition(key), key,
                    )
                    for original, derived in pairs:
                        context = ProcessingIssue("invalid_comparison_geometry", *key, original.sample_index)
                        corrected = raw_comparison_geometry(original.landmarks, original.inference_width, original.inference_height)
                        raw_error = landmark_rmse(corrected, raw_center)
                        normalized_error = landmark_rmse(derived.landmarks, normalized_center)
                        buckets[condition].add(raw_error, normalized_error, distance)
                        if condition in TARGETED_CONDITIONS:
                            targeted.add(raw_error, normalized_error, distance)
                        elif condition in POSE_CONDITIONS:
                            pose.add(raw_error, normalized_error, distance)
                        total += 1
            if total != normalized.expected_sample_count:
                raise NormalizationEvaluationError(ProcessingIssue("normalized_manifest_sample_count_mismatch"))
            context = ProcessingIssue("invalid_aggregate_metrics")
            rows = tuple(row for condition in CONDITIONS if (row := buckets[condition].summarize(condition)) is not None)
            return NormalizationQualityReport(
                len({session.manifest.config.person_id for session in sessions}), len(sessions), len(keys), total,
                rows, targeted.summarize("targeted_invariance"), pose.summarize("pose"),
            )
        except NormalizationEvaluationError:
            raise
        except Exception:
            raise NormalizationEvaluationError(context) from None
