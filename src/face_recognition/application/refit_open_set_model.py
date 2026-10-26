"""Full allowed-development refit and frozen window calibration; never holdout evaluation."""

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from face_recognition.application.calibrate_unknown_rejection import validate_rejection_data
from face_recognition.application.evaluate_temporal_open_set import FrameInference
from face_recognition.application.rejection_data import UnknownCalibrationData
from face_recognition.application.rejection_score_analysis import analyze_scores
from face_recognition.application.temporal_windows import aggregate_windows, build_windows
from face_recognition.application.training_folds import DevelopmentDataset
from face_recognition.domain.models.baseline_evaluation import EvaluationVersions
from face_recognition.domain.models.open_set_config import FrozenOpenSetConfiguration
from face_recognition.domain.models.open_set_model import OpenSetModelError, OpenSetRefitSummary


class RefitObservations(Protocol):
    parts: object
    known: FrameInference
    unknown: FrameInference
    support_vector_count: int


class RefitRunner(Protocol):
    versions: EvaluationVersions

    def fit_and_score(self, config: FrozenOpenSetConfiguration, X: np.ndarray, y: np.ndarray,
                      X_unknown: np.ndarray, class_ids: tuple[str, ...]) -> RefitObservations: ...


@dataclass(frozen=True, eq=False)
class FittedOpenSetDevelopmentModel:
    summary: OpenSetRefitSummary
    parts: object = field(repr=False)


class OpenSetModelRefitter:
    def __init__(self, runner: RefitRunner) -> None:
        self.runner = runner

    def run(self, config: FrozenOpenSetConfiguration, config_sha256: str,
            known: DevelopmentDataset, unknown: UnknownCalibrationData) -> FittedOpenSetDevelopmentModel:
        context = "Invalid authorized refit data/configuration"
        try:
            if not isinstance(config, FrozenOpenSetConfiguration) or config != FrozenOpenSetConfiguration(config.provenance, config.development_evidence):
                raise ValueError
            if len(config_sha256) != 64 or any(c not in "0123456789abcdef" for c in config_sha256):
                raise ValueError
            a = config.architecture
            validate_rejection_data(known, unknown, minimum_features=a.selected_features)
            if (known.X.shape[1] != a.original_feature_count or unknown.source.source_representation != a.source_representation
                    or len(known.samples) < config.final_threshold_calibration_policy.minimum_total_known_reference_samples):
                raise ValueError
            kp, up = build_windows(known.samples, a.temporal_window_size), build_windows(unknown.samples, a.temporal_window_size)
            if not kp.windows or not up.windows:
                raise OpenSetModelError("Complete known and unknown window-7 calibration groups are required")
            context = "Frozen full-development refit or self-excluded scoring failed"
            output = self.runner.fit_and_score(config, known.X, known.y, unknown.X, known.label_to_person)
            if (tuple(name for name, _ in output.known.confidence) != (a.novelty_method,)
                    or tuple(name for name, _ in output.unknown.confidence) != (a.novelty_method,)):
                raise ValueError
            kw = aggregate_windows(kp, output.known.predicted, output.known.decisions, output.known.confidence)
            uw = aggregate_windows(up, output.unknown.predicted, output.unknown.decisions, output.unknown.confidence)
            labels = {person: index for index, person in enumerate(known.label_to_person)}
            correct = kw.predicted == np.fromiter((labels[window.person_id] for window in kp.windows), dtype=np.int32)
            context = "Frozen window-level threshold calibration failed"
            analysis = analyze_scores(kw.confidence[0][1], correct, uw.confidence[0][1])
            summary = OpenSetRefitSummary(config_sha256, unknown.partition.source_identity_split_sha256,
                                         len(known.samples), len(labels), a.original_feature_count, a.selected_features,
                                         output.support_vector_count, kp.counts, up.counts, analysis, self.runner.versions)
            return FittedOpenSetDevelopmentModel(summary, output.parts)
        except OpenSetModelError:
            raise
        except Exception:
            raise OpenSetModelError(context) from None
