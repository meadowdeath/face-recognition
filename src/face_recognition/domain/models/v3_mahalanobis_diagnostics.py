"""Aggregate post-hoc diagnostic summaries; no window records or fitted state."""

from dataclasses import dataclass

from face_recognition.domain.models.classification import EvaluationVersions
from face_recognition.domain.models.temporal import WindowCounts
from face_recognition.domain.models.v3_mahalanobis_calibration import METHOD, THRESHOLD_TRANSPORT

CONDITION_GROUPS = (
    ("pose", ("yaw_left", "yaw_right", "pitch_up", "pitch_down", "roll_left", "roll_right")),
    ("distance", ("near", "far")),
    ("translation", ("translation_left", "translation_right", "translation_up", "translation_down")),
    ("center", ("center",)),
)


@dataclass(frozen=True)
class KnownWindowCounts:
    total_windows: int
    accepted_windows: int
    rejected_windows: int
    correct_identification_windows: int
    accepted_misidentification_windows: int
    frr: float | None
    correct_identification_rate: float | None
    accepted_misidentification_rate: float | None


@dataclass(frozen=True)
class UnknownWindowCounts:
    total_windows: int
    accepted_windows: int
    rejected_windows: int
    far: float | None
    unknown_rejection_rate: float | None


@dataclass(frozen=True)
class WindowScoreSummary:
    count: int
    mean: float | None
    median: float | None


@dataclass(frozen=True)
class FoldScoreSummary:
    contributing_folds: int
    mean_of_fold_means: float | None
    std_of_fold_means: float | None
    mean_of_fold_medians: float | None
    std_of_fold_medians: float | None


@dataclass(frozen=True)
class KnownDiagnostic:
    label: str
    counts: KnownWindowCounts
    scores: WindowScoreSummary


@dataclass(frozen=True)
class UnknownDiagnostic:
    label: str
    counts: UnknownWindowCounts
    scores: WindowScoreSummary


@dataclass(frozen=True)
class AggregateKnownDiagnostic:
    label: str
    counts: KnownWindowCounts
    scores: FoldScoreSummary


@dataclass(frozen=True)
class AggregateUnknownDiagnostic:
    label: str
    counts: UnknownWindowCounts
    scores: FoldScoreSummary


@dataclass(frozen=True)
class ErrorContribution:
    label: str
    error_windows: int
    fraction_of_population_errors: float | None
    fraction_of_population_windows: float | None


@dataclass(frozen=True)
class SessionConditionDiagnostics:
    session_id: str
    conditions: tuple[KnownDiagnostic, ...]


@dataclass(frozen=True)
class MahalanobisFoldDiagnostics:
    fold_name: str
    saved_threshold: float
    known_counts: WindowCounts
    unknown_counts: WindowCounts
    known_per_condition: tuple[KnownDiagnostic, ...]
    unknown_per_condition: tuple[UnknownDiagnostic, ...]
    known_condition_groups: tuple[KnownDiagnostic, ...]
    unknown_condition_groups: tuple[UnknownDiagnostic, ...]
    known_per_session: tuple[KnownDiagnostic, ...]
    unknown_per_identity: tuple[UnknownDiagnostic, ...]
    known_rejection_contributions: tuple[ErrorContribution, ...]
    unknown_false_acceptance_contributions: tuple[ErrorContribution, ...]
    largest_known_rejection_conditions: tuple[str, ...]
    largest_unknown_false_acceptance_identities: tuple[str, ...]
    saved_model_state_verified: bool = False
    saved_outer_summaries_verified: bool = False


@dataclass(frozen=True)
class MahalanobisAggregateDiagnostics:
    known_windows: KnownWindowCounts
    unknown_windows: UnknownWindowCounts
    known_condition_severity: tuple[AggregateKnownDiagnostic, ...]
    unknown_condition_severity: tuple[AggregateUnknownDiagnostic, ...]
    known_group_severity: tuple[AggregateKnownDiagnostic, ...]
    unknown_group_severity: tuple[AggregateUnknownDiagnostic, ...]
    known_session_severity: tuple[KnownDiagnostic, ...]
    unknown_identity_severity: tuple[UnknownDiagnostic, ...]
    known_condition_by_session: tuple[SessionConditionDiagnostics, ...]


@dataclass(frozen=True)
class MahalanobisDiagnosticProvenance:
    protocol_reference: str
    protocol_sha256: str
    protocol_file_sha256: str
    evaluation_report_reference: str
    evaluation_report_sha256: str
    initial_selection_report_sha256: str
    catalog_sha256: str
    processing_manifest_sha256: str
    payload_sha256: str
    configuration_sha256: str


@dataclass(frozen=True)
class V3MahalanobisDiagnosticsReport:
    provenance: MahalanobisDiagnosticProvenance
    folds: tuple[MahalanobisFoldDiagnostics, ...]
    aggregate: MahalanobisAggregateDiagnostics
    interpretation: tuple[str, ...]
    versions: EvaluationVersions
    schema_version: int = 1
    experiment: str = "v3_mahalanobis_diagnostics_v1"
    completion_state: str = "completed"
    method: str = METHOD
    threshold_transport: str = THRESHOLD_TRANSPORT
    scope: str = "Post-hoc diagnostics only; no model/strategy selection or calibration decisions changed; no new final performance claim"
    methodology: str = "V3 remains a cross-validation-based calibration refinement; V2 remains the last independent external holdout"
    replay_policy: str = "Models/observations were not persisted. Reconstruct identical fit states, require saved model digests and outer summaries, apply saved thresholds; no calibration or selection"
    aggregation_policy: str = "Pool disjoint outer-test decision counts. Never pool raw scores across fold models; report unweighted mean/population std of fold means/medians instead. Each known test session/UNKNOWN identity belongs to one fold."
    ranking_policy: str = "Severity: higher FRR/FAR, then more errors, then lexical label; missing rates last. Contribution: more error windows then lexical label; retain all largest-count ties."
    replay_limitation: str = "Attributions are reconstructed, not extracted from retained individual observations. Reproduction verifies saved learned state, frozen-threshold/closed-set metrics, counts and score distributions."
