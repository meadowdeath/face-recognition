"""Aggregate metadata for the fixed-method, same-model V3 calibration study."""

from dataclasses import dataclass, field

from face_recognition.domain.models.classification import ClassificationMetrics, EvaluationVersions
from face_recognition.domain.models.rejection import RejectionMetrics
from face_recognition.domain.models.score_analysis import ScoreSeparation
from face_recognition.domain.models.temporal import WindowCounts
from face_recognition.domain.models.v3_novelty_evaluation import BinaryCalibrationMetrics, V3NoveltyConfiguration

_BASE = V3NoveltyConfiguration()
METHOD = "mahalanobis_distance"
THRESHOLD_TRANSPORT = "none_same_model_calibration"


@dataclass(frozen=True)
class V3MahalanobisConfiguration:
    representation: str = _BASE.representation
    input_features: int = _BASE.input_features
    selected_features: int = _BASE.selected_features
    selector: str = _BASE.selector
    scaler: str = _BASE.scaler
    classifier: str = _BASE.classifier
    svm_parameters: tuple = _BASE.svm_parameters
    covariance: str = _BASE.covariance
    covariance_ridge_relative: float = _BASE.covariance_ridge_relative
    covariance_ridge_rule: str = _BASE.covariance_ridge_rule
    novelty_method: str = METHOD
    window_size: int = _BASE.window_size
    confidence_aggregation: str = _BASE.confidence_aggregation
    identity_aggregation: str = _BASE.identity_aggregation
    consistency_gate: bool = False
    threshold_rule: str = _BASE.threshold_rule
    threshold_transport: str = THRESHOLD_TRANSPORT
    known_role_policy: str = "Next globally ordered session cyclically after outer test calibrates; other two sessions fit"
    unknown_role_policy: str = "All six outer-development identities calibrate; two outer-test identities score only after threshold freeze; both sessions stay together"


@dataclass(frozen=True)
class CyclicFoldRoles:
    fold_name: str
    known_fit_sessions: tuple[str, ...]
    known_calibration_session: str
    known_test_session: str
    unknown_calibration_ids: tuple[str, ...]
    unknown_test_ids: tuple[str, ...]


@dataclass(frozen=True)
class SelectionReportProvenance:
    report_sha256: str
    protocol_sha256: str
    catalog_sha256: str
    payload_sha256: str
    configuration_sha256: str
    identity_count: int
    source_samples: int
    selected_methods: tuple[tuple[str, str], ...]
    transport_warning_folds: int
    schema_version: int = 1
    experiment: str = "v3_nested_novelty_comparison_v1"
    usage: str = "Method-selection provenance only; prior thresholds and performance metrics are not calibration inputs"


@dataclass(frozen=True)
class MahalanobisCalibrationFold:
    roles: CyclicFoldRoles
    fit_known_frames: int
    threshold: float
    known_calibration_counts: WindowCounts
    unknown_calibration_counts: WindowCounts
    calibration_metrics: BinaryCalibrationMetrics
    calibration_diagnostics: ScoreSeparation
    known_test_counts: WindowCounts
    unknown_test_counts: WindowCounts
    closed_set_known_frames: ClassificationMetrics
    operational_metrics: RejectionMetrics
    diagnostics: ScoreSeparation
    model_sha256_before_calibration: str
    model_sha256_after_test: str
    fit_calls: int
    same_model_identity_verified: bool
    no_refit_after_calibration: bool
    threshold_transport: str = THRESHOLD_TRANSPORT


@dataclass(frozen=True)
class V3MahalanobisCalibrationReport:
    protocol_sha256: str
    protocol_file_sha256: str
    catalog_sha256: str
    processing_manifest_sha256: str
    payload_sha256: str
    configuration_sha256: str
    selection_provenance: SelectionReportProvenance
    identity_count: int
    source_samples: int
    outer_results: tuple[MahalanobisCalibrationFold, ...]
    mean_closed_set: ClassificationMetrics
    std_closed_set: ClassificationMetrics
    mean_metrics: RejectionMetrics
    std_metrics: RejectionMetrics
    mean_roc_auc: float
    std_roc_auc: float
    mean_eer: float
    std_eer: float
    far_range: tuple[float, float]
    frr_range: tuple[float, float]
    versions: EvaluationVersions
    configuration: V3MahalanobisConfiguration = field(default_factory=V3MahalanobisConfiguration)
    schema_version: int = 1
    experiment: str = "v3_mahalanobis_same_model_calibration_v1"
    completion_state: str = "completed"
    threshold_transport: str = THRESHOLD_TRANSPORT
    methodology: str = "V3 threshold-calibration refinement under cross-validation; prior outer results were inspected and motivated this revision"
    limitation: str = "Not pristine untouched final evaluation or independent final generalization. V3 has no additional untouched holdout; V2 remains the last independent external holdout."
    aggregation: str = "Unweighted four-fold means and population std; FAR/FRR min/max; never pool raw score distributions across folds"
