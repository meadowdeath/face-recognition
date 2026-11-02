"""Aggregate evidence for the controlled, iterative V3 representation comparison."""

from dataclasses import dataclass

from face_recognition.domain.models.classification import ClassificationMetrics, EvaluationVersions
from face_recognition.domain.models.rejection import RejectionMetrics
from face_recognition.domain.models.v3_mahalanobis_calibration import MahalanobisCalibrationFold, V3MahalanobisConfiguration
from face_recognition.domain.models.v3_mahalanobis_diagnostics import KnownDiagnostic, UnknownDiagnostic, MahalanobisAggregateDiagnostics

POSE_DELTA_CONDITIONS = ("yaw_left", "yaw_right", "pitch_up", "pitch_down", "roll_left", "roll_right", "near", "far")


@dataclass(frozen=True)
class PoseRepresentationFold:
    evaluation: MahalanobisCalibrationFold
    known_per_condition: tuple[KnownDiagnostic, ...]
    unknown_per_condition: tuple[UnknownDiagnostic, ...]
    known_per_session: tuple[KnownDiagnostic, ...]
    unknown_per_identity: tuple[UnknownDiagnostic, ...]


@dataclass(frozen=True)
class RepresentationResult:
    representation: str
    catalog_sha256: str
    processing_manifest_sha256: str
    payload_sha256: str
    configuration: V3MahalanobisConfiguration
    configuration_sha256: str
    folds: tuple[PoseRepresentationFold, ...]
    mean_closed_set: ClassificationMetrics
    std_closed_set: ClassificationMetrics
    mean_metrics: RejectionMetrics
    std_metrics: RejectionMetrics
    mean_roc_auc: float
    std_roc_auc: float
    mean_eer: float
    std_eer: float
    diagnostics: MahalanobisAggregateDiagnostics
    versions: EvaluationVersions


@dataclass(frozen=True)
class ConditionRepresentationComparison:
    condition: str
    baseline_frr: float
    candidate_frr: float
    frr_delta_candidate_minus_baseline: float
    baseline_correct_identification_rate: float
    candidate_correct_identification_rate: float


@dataclass(frozen=True)
class V3PoseRepresentationReport:
    protocol_sha256: str
    protocol_file_sha256: str
    baseline_protocol_catalog_sha256: str
    normalization_definition: dict
    normalization_definition_sha256: str
    results: tuple[RepresentationResult, ...]
    condition_comparison: tuple[ConditionRepresentationComparison, ...]
    pose_distance_frr_deltas: tuple[tuple[str, float], ...]
    ranking: tuple[str, ...]
    selected_development_representation: str
    schema_version: int = 1
    experiment: str = "v3_pose_representation_comparison_v1"
    completion_state: str = "completed"
    threshold_transport: str = "none_same_model_calibration"
    selection_rule: str = "Higher mean outer binary balanced accuracy; lower mean FRR; lower mean FAR; higher mean closed-set macro F1; all_xyz_v1 wins exact tie"
    methodology: str = "Iterative V3 representation experiment motivated by inspected post-hoc diagnostics; not independent final generalization. V2 remains the last independent external holdout."
    calibration: str = "Same cyclic fit/calibration/test roles for both representations. Fit two KNOWN sessions; calibrate on the next session plus all six development UNKNOWN identities; never refit after threshold calibration."
    aggregation: str = "Unweighted outer-fold mean/population std, including AUC/EER. Pool disjoint decision counts for condition diagnostics; never pool raw scores across fold models."
    privacy: str = "Aggregate results only; no individual features, scores, predictions or fitted biometric parameters"
