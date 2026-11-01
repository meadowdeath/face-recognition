"""Independent V2 final authorization and aggregate results; never sample values."""

from dataclasses import dataclass

from face_recognition.domain.models.baseline_evaluation import EvaluationVersions
from face_recognition.domain.models.final_evaluation import ComparisonValues, FinalClosedSet, FinalConditionMetrics, FinalPartitionCounts
from face_recognition.domain.models.rejection_score_evaluation import ScoreSeparation
from face_recognition.domain.models.temporal_open_set import VoteStability, WindowCounts
from face_recognition.domain.models.unknown_rejection import RejectionMetrics
from face_recognition.domain.models.v2_open_set_config import FrozenV2OpenSetConfiguration

V2_NO_RETUNING = "Final V2 holdout results must not be used to modify or retune this V2 model version."


@dataclass(frozen=True, order=True)
class FinalV2Session:
    person_id: str
    session_id: str


@dataclass(frozen=True)
class FinalV2Condition:
    person_id: str
    session_id: str
    condition: str


@dataclass(frozen=True)
class FinalV2HoldoutManifest:
    frozen_config_sha256: str
    development_report_sha256: str
    development_identity_split_sha256: str
    development_unknown_split_sha256: str
    model_manifest_sha256: str
    processing_manifest_sha256: str
    known_sessions: tuple[FinalV2Session, ...]
    unknown_sessions: tuple[FinalV2Session, ...]
    condition_files: tuple[FinalV2Condition, ...]
    expected_total_samples: int
    schema_version: int = 1
    manifest_id: str = "final_holdout_v2"
    representation: str = "all_xyz_v1"
    original_feature_count: int = 1434
    completion_state: str = "catalog_validated"
    validation_scope: str = "Catalog/processing metadata and independent role authorization ONLY; no JSONL opened or feature values inspected. Every record is validated before final inference."
    ordering: str = "Person/session lexical, guided capture condition order, then sample_index"
    purpose: str = "Independent final V2 evaluation only; no development, fitting or calibration"


@dataclass(frozen=True)
class FinalV2Sample:
    person_id: str
    session_id: str
    condition: str
    sample_index: int
    source_relative_path: str


@dataclass(frozen=True)
class FinalV2Context:
    configuration: FrozenV2OpenSetConfiguration
    frozen_config_sha256: str
    development_report_sha256: str
    development_identity_split_sha256: str
    development_unknown_split_sha256: str
    final_holdout_manifest_sha256: str
    processing_manifest_sha256: str
    model_fingerprints: tuple[tuple[str, str], ...]
    threshold: float
    calibration_roc_auc: float
    calibration_balanced_accuracy: float
    model_id: str = "open_set_v2"
    model_status: str = "fitted_and_calibrated"


@dataclass(frozen=True)
class FinalV2EvaluationReport:
    context: FinalV2Context
    known_final: FinalPartitionCounts
    unknown_final: FinalPartitionCounts
    closed_set_known_frames: FinalClosedSet
    frame_score_diagnostics: ScoreSeparation
    known_windows: WindowCounts
    unknown_windows: WindowCounts
    operational_metrics: RejectionMetrics
    temporal_score_diagnostics: ScoreSeparation
    conditions: tuple[FinalConditionMetrics, ...]
    known_vote_stability: VoteStability
    development_candidate: ComparisonValues
    final_holdout: ComparisonValues
    delta_final_minus_development: ComparisonValues
    evaluation_versions: EvaluationVersions
    closed_set_delta_final_minus_development: tuple[tuple[str, float], ...]
    schema_version: int = 1
    experiment: str = "final_v2_open_set_evaluation_v1"
    status: str = "final_holdout_evaluation_completed"
    scope: str = "Independent final V2 generalization evaluation; frame closed-set and temporal open-set metrics are distinct"
    comparison_scope: str = "Development selection and OOF calibration are NOT final generalization performance. Descriptive comparison only; no retuning."
    frozen_snapshot_scope: str = "Embedded configuration is the immutable pre-test freeze snapshot; its no-fit/no-final-evaluation flags describe that historical boundary, not this completed final report."
    diagnostic_threshold_policy: str = "ROC/EER thresholds are diagnostics only; ALL operational decisions use the exact persisted model threshold"
    no_retuning: str = V2_NO_RETUNING
    fitting_performed: bool = False
    threshold_calibration_performed: bool = False
    model_artifacts_unchanged: bool = True
    final_known_payloads_accessed: bool = True
    final_unknown_payloads_accessed: bool = True
