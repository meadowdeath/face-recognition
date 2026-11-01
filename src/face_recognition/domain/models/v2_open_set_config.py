"""Frozen V2 choices and pre-test procedure, never fitted biometric parameters."""

from dataclasses import dataclass, field
from math import isfinite
import re

from face_recognition.domain.models.baseline_evaluation import ClassificationMetrics
from face_recognition.domain.models.dataset import validate_identifier
from face_recognition.domain.models.open_set_config import (
    ConfigurationFreezeError, DevelopmentWindowPolicy, FrozenMetricDefinitions, OpenSetArchitecture,
)
from face_recognition.domain.models.unknown_rejection import RejectionMetrics
from face_recognition.domain.models.v2_diagnostics import StudyFold

V2_FROZEN_CANDIDATE = "svm_top1_global__window_7__normal"
V2_DEVELOPMENT_EXPERIMENT = "v2_open_set_diagnostics_v1"


@dataclass(frozen=True)
class V2Architecture:
    source_representation: str = "all_xyz_v1"
    original_feature_count: int = 1434
    feature_selection_method: str = "SelectKBest"
    feature_selection_score: str = "f_classif"
    selected_features: int = 448
    scaler_type: str = "StandardScaler"
    scaler_parameters: tuple = OpenSetArchitecture().scaler_parameters
    classifier_type: str = "SVC"
    svm_parameters: tuple = OpenSetArchitecture().svm_parameters
    decision_function_shape: str = "ovr"
    svc_break_ties: bool = False
    rejection_strategy: str = "svm_top1_global"
    rejection_score_family: str = "svm_top1"
    rejection_threshold_type: str = "global"
    frame_score_definition: str = "Maximum SVM class decision_function score in deterministic OVR class order; binary signed d expands to [-d,d]"
    score_orientation: str = "higher means more likely KNOWN"
    temporal_window_size: int = 7
    rejection_temporal_aggregation: str = "median"
    identity_temporal_aggregation: str = "majority_vote"
    identity_tie_breaking: tuple = OpenSetArchitecture().identity_tie_breaking
    identity_consistency_gate: bool = False
    class_order: str = OpenSetArchitecture().class_order
    decision_rule: str = "Accept aggregated SVM identity when median of seven svm_top1 frame scores >= global threshold; otherwise UNKNOWN"


@dataclass(frozen=True)
class V2ThresholdPolicy:
    policy_version: str = "cross_session_oof_svm_top1_window_calibration_v2"
    allowed_data_sources: tuple[str, ...] = ("ORIGINAL known_train_pool", "ORIGINAL known_test",
                                            "ORIGINAL unknown_calibration", "ORIGINAL unknown_holdout")
    source_boundary: str = "Only original V1 identities/sessions bound to provenance.original_identity_split_sha256; all are now V2 DEVELOPMENT"
    prohibited_data_sources: tuple[str, ...] = ("new known final-test session (conceptually session_04)",
                                               "new UNKNOWN final-test identities")
    known_calibration: str = "Pool held-out-session window scores from the three recorded folds; every known window occurs once; no sample is scored by a model fitted on its session"
    unknown_calibration: str = "ALL current original UNKNOWN development identities, scored under EACH of the same three fold models; pool one score per unknown window per model"
    unknown_observation_accounting: str = "Each unknown window contributes three model observations; known windows contribute one OOF observation. Binary balanced accuracy gives equal weight to the two roles regardless of population counts. This is calibration, not independent validation."
    candidate_selection_history: str = "Identity-level UNKNOWN rotations selected the strategy in development; final pre-test numeric calibration then uses ALL original development UNKNOWN identities"
    ordered_steps: tuple[str, ...] = (
        "For each recorded fold, fit ANOVA-448, StandardScaler and fixed SVC using only the two known fit sessions",
        "Score the held-out known session and ALL current development UNKNOWN identities with that fold's fitted pipeline",
        "Within each role and fold build frozen non-overlapping window-7 groups; median svm_top1 score",
        "Pool the three known OOF window populations and the three UNKNOWN model-observation populations",
        "Run the existing deterministic threshold search and balanced-accuracy tie-breaking rule on those pooled scores",
        "Record per-fold score distributions and thresholds as DEVELOPMENT stability diagnostics, never substitute one fold threshold",
        "Refit the identity pipeline on ALL three original known development sessions, without fitting on UNKNOWN data",
        "Directly transfer the pooled OOF threshold to that final all-session model; no resubstitution recalibration or automatic alignment",
        "Persist and freeze the final model and finite numeric threshold BEFORE collecting/accessing new final-test data",
    )
    same_training_pool_resubstitution: bool = False
    copy_fold_threshold: bool = False
    score_scale_alignment: str = "none; direct OOF threshold transfer"
    transfer_limitation: str = "SVM scores may shift after all-session refit. The next step must record fold score/threshold stability and assess direct transfer using DEVELOPMENT diagnostics only. If alignment is necessary, stop and explicitly predeclare a new development-only policy/version BEFORE collecting/accessing future data; never adjust using final results."
    threshold_search: str = "face_recognition.application.rejection_metrics.calibrate_threshold with allow_negative=True"
    threshold_candidates: str = "Sorted unique observed scores, finite interior midpoints, accept-all min(0,min_score), finite nextafter(max_score,+inf) reject-all boundary"
    selection_rule: tuple[str, ...] = ("higher binary balanced accuracy", "lower FAR", "lower FRR", "higher threshold")
    diagnostic_far_targets: tuple[float, ...] = (.20, .10, .05)
    diagnostic_far_policy: str = "Diagnostics only; never replace the balanced-accuracy threshold or choose a production FAR target from final-test results"
    decision_rule: str = "median_svm_top1_score >= threshold means KNOWN; otherwise UNKNOWN"


@dataclass(frozen=True)
class V2FutureTestPolicy:
    known_test: str = "New known session absent from V2 development (conceptually session_04), same registered known identities and guided protocol where feasible"
    unknown_test: str = "New UNKNOWN participants never used in V2 development or calibration"
    unknown_count_guidance: str = "Approximately three preserves V1 comparability; more improves breadth; not an algorithm requirement"
    configuration_freeze_before_collection: bool = True
    fitted_model_and_threshold_freeze_before_collection: bool = True
    future_data_accessed: bool = False
    final_evaluation_performed: bool = False
    no_retuning: str = "Future final-test results may evaluate this version only; must not modify its configuration, pipeline or threshold"


@dataclass(frozen=True)
class V2ReportProvenance:
    sha256: str
    v1_final_report_sha256: str
    original_identity_split_sha256: str
    experiment: str = field(default=V2_DEVELOPMENT_EXPERIMENT, init=False)
    schema_version: int = field(default=1, init=False)
    status: str = field(default="development_diagnostics_completed", init=False)
    selected_candidate: str = field(default=V2_FROZEN_CANDIDATE, init=False)
    digest_scope: str = field(default="Exact aggregate V2 report bytes; no source path stored", init=False)

    def __post_init__(self):
        for digest in (self.sha256, self.v1_final_report_sha256, self.original_identity_split_sha256):
            if not isinstance(digest, str) or re.fullmatch("[0-9a-f]{64}", digest) is None:
                raise ConfigurationFreezeError("Invalid V2 configuration provenance digest")


@dataclass(frozen=True)
class V2DevelopmentEvidence:
    closed_set_frame_metrics: ClassificationMetrics
    mean_roc_auc: float
    mean_eer: float
    metrics: RejectionMetrics
    known_source_samples: int
    unknown_source_samples: int
    known_identity_count: int
    unknown_identity_count: int
    known_unique_windows: int
    unknown_three_model_window_observations: int
    scope: str = field(default="V2 DEVELOPMENT evidence only; known thresholds calibrated/described out of session, UNKNOWN identity rotations; not final generalization performance", init=False)

    def __post_init__(self):
        if not isinstance(self.closed_set_frame_metrics, ClassificationMetrics) or not isinstance(self.metrics, RejectionMetrics):
            raise ConfigurationFreezeError("Invalid V2 development evidence")
        rates = (self.mean_roc_auc, self.mean_eer, *vars(self.metrics).values())
        if any(type(v) not in (float, int) or not isfinite(v) or not 0 <= v <= 1 for v in rates):
            raise ConfigurationFreezeError("V2 evidence metrics must be finite rates")
        counts = (self.known_source_samples, self.unknown_source_samples, self.known_identity_count,
                  self.unknown_identity_count, self.known_unique_windows, self.unknown_three_model_window_observations)
        if any(type(v) is not int or v < 1 for v in counts):
            raise ConfigurationFreezeError("V2 evidence counts must be positive integers")


@dataclass(frozen=True)
class FrozenV2OpenSetConfiguration:
    provenance: V2ReportProvenance
    development_evidence: V2DevelopmentEvidence
    known_cross_session_folds: tuple[StudyFold, ...]
    schema_version: int = field(default=1, init=False)
    configuration_id: str = field(default="open_set_config_v2", init=False)
    status: str = field(default="development_configuration_frozen", init=False)
    completion_state: str = field(default="frozen", init=False)
    architecture: V2Architecture = field(default=V2Architecture(), init=False)
    development_window_policy: DevelopmentWindowPolicy = field(default=DevelopmentWindowPolicy(), init=False)
    final_threshold_calibration_policy: V2ThresholdPolicy = field(default=V2ThresholdPolicy(), init=False)
    future_final_test: V2FutureTestPolicy = field(default=V2FutureTestPolicy(), init=False)
    metric_definitions: FrozenMetricDefinitions = field(default=FrozenMetricDefinitions(
        version="open_set_window_and_closed_frame_metrics_v2",
        unit="Closed-set identity uses known frames; rejection/open-set uses complete windows",
        definitions=tuple((key, value.replace("known windows BEFORE", "known frames BEFORE"))
                          for key, value in FrozenMetricDefinitions().definitions),
        development_aggregation="Source study uses mean AUC/EER over session/UNKNOWN rotations and role-adjusted operating metrics; future numeric calibration pools known OOF and UNKNOWN three-model windows",
    ), init=False)
    final_numeric_threshold: None = field(default=None, init=False)
    final_threshold_status: str = field(default="not_numerically_calibrated; cross_session_policy_frozen", init=False)
    final_model_fitted: bool = field(default=False, init=False)
    v2_final_evaluation_performed: bool = field(default=False, init=False)
    biometric_payloads_accessed: bool = field(default=False, init=False)
    v1_immutable: bool = field(default=True, init=False)
    mutation_policy: str = field(default="No overwrite/unfreeze or routine force option; different choices require a new version and explicit development reasoning", init=False)

    def __post_init__(self):
        if not isinstance(self.provenance, V2ReportProvenance) or not isinstance(self.development_evidence, V2DevelopmentEvidence):
            raise ConfigurationFreezeError("Frozen V2 configuration needs validated provenance and evidence")
        folds = self.known_cross_session_folds
        if not isinstance(folds, tuple) or len(folds) != 3 or any(not isinstance(f, StudyFold) for f in folds):
            raise ConfigurationFreezeError("Exactly three known-session folds are required")
        sessions = tuple(f.calibration_session for f in reversed(folds))
        for session in sessions:
            validate_identifier(session)
        if len(set(sessions)) != 3 or any(f.name != f"Fold {name}" or f.fit_sessions != tuple(s for s in sessions if s != f.calibration_session)
                                         for name, f in zip("ABC", folds)):
            raise ConfigurationFreezeError("Inconsistent V2 known cross-session folds")
