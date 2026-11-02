"""Immutable V3 operational decisions and aggregate evidence, never fitted state."""

from dataclasses import dataclass, field

from face_recognition.domain.models.processed_sample import POSE_REPRESENTATION
from face_recognition.domain.models.v3_mahalanobis_calibration import V3MahalanobisConfiguration

_CURRENT = V3MahalanobisConfiguration()
EVIDENCE_KINDS = ("protocol", "novelty_report", "calibration_report", "diagnostics_report", "pose_report")
EVIDENCE_SOURCE_TYPES = ("v3_nested_session_identity_cv", "v3_nested_novelty_comparison_v1",
    "v3_mahalanobis_same_model_calibration_v1", "v3_mahalanobis_diagnostics_v1", "v3_pose_representation_comparison_v1")
IMPLEMENTATION_MODULES = (
    "face_recognition.infrastructure.features.pose_landmark_normalizer",
    "face_recognition.infrastructure.features.mahalanobis",
)
LIMITATIONS = (
    "Yaw remains difficult despite improvement with pose-relative coordinates.",
    "pitch_up did not improve in the representation comparison.",
    "pitch_down remained unchanged in the representation comparison.",
    "Session/context variation remains; no causal lighting attribution is available.",
    "Some development UNKNOWN identities were harder to reject; prior diagnostics highlighted pseudonymous identity p009.",
    "MediaPipe z is relative depth, not metric 3D reconstruction or true metric frontalization.",
)


@dataclass(frozen=True)
class FrozenPoseNormalization:
    version: str
    landmark_count: int
    first_eye: tuple[int, ...]
    second_eye: tuple[int, ...]
    superior_reference: tuple[int, ...]
    inferior_reference: tuple[int, ...]
    aspect_correction: str
    origin: str
    scale: str
    horizontal: str
    vertical: str
    depth: str
    minimum_length: float
    minimum_axis_sine: float
    limitation: str


@dataclass(frozen=True)
class FrozenV3Pipeline:
    representation: str = POSE_REPRESENTATION
    input_features: int = _CURRENT.input_features
    selector: str = _CURRENT.selector
    selected_features: int = _CURRENT.selected_features
    scaler: str = _CURRENT.scaler
    classifier: str = _CURRENT.classifier
    svm_parameters: tuple = _CURRENT.svm_parameters
    novelty_method: str = _CURRENT.novelty_method
    covariance: str = _CURRENT.covariance
    covariance_ridge_relative: float = _CURRENT.covariance_ridge_relative
    covariance_ridge_rule: str = _CURRENT.covariance_ridge_rule
    novelty_reference: str = "Per-KNOWN-class shrinkage distribution fitted on KNOWN fit data only; score the SVM-predicted class"
    knownness_score: str = "-mahalanobis_distance"
    score_orientation: str = "higher_is_known"
    unknown_fits_model_components: bool = False


@dataclass(frozen=True)
class FrozenV3TemporalPolicy:
    window_size: int = _CURRENT.window_size
    window_policy: str = "non_overlapping"
    grouping: tuple[str, ...] = ("person_id", "session_id", "condition")
    ordering: str = "sample_index"
    incomplete_tails: str = "discard_and_count"
    score_aggregation: str = _CURRENT.confidence_aggregation
    identity_aggregation: str = _CURRENT.identity_aggregation
    consistency_gate: bool = False


@dataclass(frozen=True)
class FrozenV3ThresholdPolicy:
    threshold_transport: str = "none_same_model_calibration"
    same_fitted_state_required: bool = True
    fitted_components: tuple[str, ...] = ("selector", "scaler", "identity_svm", "mahalanobis_references_and_covariances")
    post_calibration_refit_allowed: bool = False
    calibration_level: str = "window_7_median_knownness"
    objective: str = "maximize_binary_balanced_accuracy"
    tie_breakers: tuple[str, ...] = ("lower_unknown_FAR", "lower_known_FRR", "higher_numeric_threshold")
    enumeration: str = _CURRENT.threshold_rule
    acceptance_rule: str = "window_knownness_score >= threshold"
    auc_eer_role: str = "diagnostic_only"
    deployable_population_allocation: str = "not_defined_in_this_configuration; requires a subsequent explicit policy"
    numeric_threshold_status: str = "not_calibrated"
    numeric_threshold: None = None


@dataclass(frozen=True)
class FrozenEvidenceReference:
    kind: str
    filename: str
    sha256: str
    schema_version: int
    source_type: str


@dataclass(frozen=True)
class ImplementationFingerprint:
    module: str
    sha256: str


@dataclass(frozen=True)
class DevelopmentMetricEvidence:
    source: str
    metric: str
    value: float


@dataclass(frozen=True)
class FrozenV3OpenSetConfiguration:
    normalization: FrozenPoseNormalization
    normalization_sha256: str
    implementations: tuple[ImplementationFingerprint, ...]
    evidence: tuple[FrozenEvidenceReference, ...]
    protocol_sha256: str
    protocol_catalog_sha256: str
    development_evidence: tuple[DevelopmentMetricEvidence, ...]
    pipeline: FrozenV3Pipeline = field(default_factory=FrozenV3Pipeline)
    temporal: FrozenV3TemporalPolicy = field(default_factory=FrozenV3TemporalPolicy)
    threshold_policy: FrozenV3ThresholdPolicy = field(default_factory=FrozenV3ThresholdPolicy)
    limitations: tuple[str, ...] = LIMITATIONS
    schema_version: int = 1
    configuration_id: str = "open_set_config_v3"
    status: str = "development_configuration_frozen"
    completion_state: str = "frozen"
    configuration_sha256: str = ""
    study_status: str = "Iterative cross-validation-based V3 development study"
    independent_untouched_holdout_available: bool = False
    last_independent_external_holdout: str = "V2"
    interpretation: str = "V3 results must not be described as independent final generalization"
    deployable_model_status: str = "not_fitted_by_freeze_workflow"
    mutation_policy: str = "No overwrite or unfreeze; changed decisions require a new configuration version and explicit development protocol"
