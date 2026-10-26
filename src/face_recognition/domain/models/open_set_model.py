"""Non-sensitive fitted-model summary; biometric fitted parts remain infrastructure."""

from dataclasses import dataclass
from math import isfinite

from face_recognition.domain.models.baseline_evaluation import EvaluationVersions
from face_recognition.domain.models.rejection_score_evaluation import ScoreAnalysis
from face_recognition.domain.models.temporal_open_set import WindowCounts


class OpenSetModelError(ValueError):
    """Sanitized refit, calibration or artifact-integrity error."""


@dataclass(frozen=True)
class OpenSetRefitSummary:
    frozen_config_sha256: str
    source_identity_split_sha256: str
    known_training_samples: int
    identity_count: int
    original_feature_count: int
    selected_feature_count: int
    support_vector_count: int
    known_calibration: WindowCounts
    unknown_calibration: WindowCounts
    analysis: ScoreAnalysis
    versions: EvaluationVersions
    known_test_payloads_accessed: bool = False
    unknown_holdout_payloads_accessed: bool = False
    final_holdouts_evaluated: bool = False
    scope: str = "Development-only full refit and window-level calibration; known calibration is training resubstitution with neighbor self-exclusion, NOT independent generalization accuracy"

    def __post_init__(self) -> None:
        for digest in (self.frozen_config_sha256, self.source_identity_split_sha256):
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise OpenSetModelError("Invalid refit provenance digest")
        values = (self.known_training_samples, self.identity_count, self.original_feature_count,
                  self.selected_feature_count, self.support_vector_count)
        if (any(type(value) is not int or value < 1 for value in values) or self.identity_count < 2
                or self.support_vector_count > self.known_training_samples or self.selected_feature_count > self.original_feature_count):
            raise OpenSetModelError("Invalid refit structural counts")
        threshold = self.analysis.calibration.threshold
        if type(threshold) not in (float, int) or not isfinite(threshold):
            raise OpenSetModelError("Calibrated threshold must be finite")
        if (self.analysis.known_distribution.count != self.known_calibration.windows
                or self.analysis.unknown_distribution.count != self.unknown_calibration.windows
                or self.known_calibration.source_samples != self.known_training_samples
                or self.known_calibration.windows < 1 or self.unknown_calibration.windows < 1):
            raise OpenSetModelError("Calibration must use complete temporal window populations")
        if self.known_test_payloads_accessed or self.unknown_holdout_payloads_accessed or self.final_holdouts_evaluated:
            raise OpenSetModelError("Protected holdout use is forbidden during refit/calibration")
