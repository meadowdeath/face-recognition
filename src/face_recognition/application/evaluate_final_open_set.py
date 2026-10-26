"""One frozen operational threshold; holdout EER is descriptive, never calibration."""

from dataclasses import fields
from math import isfinite
from typing import Protocol

import numpy as np

from face_recognition.application.evaluate_temporal_open_set import FrameInference
from face_recognition.application.final_evaluation_data import FinalEvaluationError, FinalHoldoutData
from face_recognition.application.rejection_metrics import evaluate_threshold
from face_recognition.application.rejection_score_analysis import score_separation
from face_recognition.application.temporal_windows import aggregate_windows, build_windows
from face_recognition.domain.models.baseline_evaluation import EvaluationVersions
from face_recognition.domain.models.dataset import CONDITIONS
from face_recognition.domain.models.final_evaluation import (
    ComparisonValues, FinalClosedSet, FinalConditionMetrics, FinalEvaluationReport, FinalModelContext, FinalPartitionCounts,
)


class FinalInference(Protocol):
    known: FrameInference
    unknown: FrameInference
    closed_set: FinalClosedSet


class FinalRunner(Protocol):
    versions: EvaluationVersions
    frozen_threshold: float
    class_ids: tuple[str, ...]

    def infer(self, known_X: np.ndarray, known_y: np.ndarray, unknown_X: np.ndarray) -> FinalInference: ...


class FinalOpenSetEvaluator:
    def __init__(self, runner: FinalRunner):
        self.runner = runner

    def evaluate(self, data: FinalHoldoutData, context: FinalModelContext) -> FinalEvaluationReport:
        try:
            # Revalidate even when called without the filesystem loader.
            FinalHoldoutData(data.known_X, data.known_y, data.known_samples, data.unknown_X, data.unknown_samples,
                             data.class_ids, data.source, data.partition)
            a = context.configuration.architecture
            if (data.known_X.shape[1] != a.original_feature_count or type(context.threshold) not in (float, int)
                    or not isfinite(context.threshold) or context.threshold != self.runner.frozen_threshold or data.class_ids != self.runner.class_ids):
                raise ValueError
            kp, up = build_windows(data.known_samples, a.temporal_window_size), build_windows(data.unknown_samples, a.temporal_window_size)
            if not kp.windows or not up.windows:
                raise FinalEvaluationError("Final populations require complete frozen-window groups")
            output = self.runner.infer(data.known_X, data.known_y, data.unknown_X)
            if (tuple(name for name, _ in output.known.confidence) != (a.novelty_method,)
                    or tuple(name for name, _ in output.unknown.confidence) != (a.novelty_method,)):
                raise ValueError
            kw = aggregate_windows(kp, output.known.predicted, output.known.decisions, output.known.confidence)
            uw = aggregate_windows(up, output.unknown.predicted, output.unknown.decisions, output.unknown.confidence)
            labels = {person: index for index, person in enumerate(data.class_ids)}
            correct = kw.predicted == np.fromiter((labels[w.person_id] for w in kp.windows), dtype=np.int32)
            ks, us = kw.confidence[0][1], uw.confidence[0][1]
            # This is the ONLY operational threshold, passed unchanged from persisted model.
            metrics = evaluate_threshold(ks, correct, us, context.threshold, allow_negative=True)
            frame_diagnostics = score_separation(output.known.confidence[0][1], output.unknown.confidence[0][1])
            temporal_diagnostics = score_separation(ks, us)
            conditions = []
            known_conditions = np.asarray([s.condition for s in data.known_samples])
            unknown_conditions = np.asarray([s.condition for s in data.unknown_samples])
            for condition in CONDITIONS:
                kframes, uframes = known_conditions == condition, unknown_conditions == condition
                kmask, umask = np.asarray(kw.conditions) == condition, np.asarray(uw.conditions) == condition
                nk, nu = int(kmask.sum()), int(umask.sum())
                if not kframes.any() and not uframes.any():
                    continue
                conditions.append(FinalConditionMetrics(
                    condition, int(kframes.sum()), int(uframes.sum()), nk, nu,
                    float((output.known.predicted[kframes] == data.known_y[kframes]).mean()) if kframes.any() else None,
                    float((ks[kmask] >= context.threshold).mean()) if nk else None,
                    float((ks[kmask] < context.threshold).mean()) if nk else None,
                    float(((ks[kmask] >= context.threshold) & correct[kmask]).mean()) if nk else None,
                    float((us[umask] < context.threshold).mean()) if nu else None,
                    float((us[umask] >= context.threshold).mean()) if nu else None,
                ))
            evidence = context.configuration.development_evidence
            development = ComparisonValues(evidence.mean_roc_auc, evidence.mean_eer, evidence.mean_metrics.unknown_false_acceptance_rate,
                                           evidence.mean_metrics.known_rejection_rate, evidence.mean_metrics.known_correct_identification_rate,
                                           evidence.mean_metrics.open_set_exact_accuracy)
            final = ComparisonValues(temporal_diagnostics.roc_auc, temporal_diagnostics.equal_error_point.eer,
                                     metrics.unknown_false_acceptance_rate, metrics.known_rejection_rate,
                                     metrics.known_correct_identification_rate, metrics.open_set_exact_accuracy)
            delta = ComparisonValues(*(getattr(final, f.name) - getattr(development, f.name) for f in fields(ComparisonValues)))

            def counts(samples):
                return FinalPartitionCounts(len({s.person_id for s in samples}), len({(s.person_id, s.session_id) for s in samples}),
                                            len({s.source_relative_path for s in samples}), len(samples))
            return FinalEvaluationReport(context, data.partition.source_identity_split_sha256, counts(data.known_samples), counts(data.unknown_samples),
                                         output.closed_set, frame_diagnostics, kp.counts, up.counts, metrics, temporal_diagnostics,
                                         tuple(conditions), kw.vote_stability, development, final, delta, self.runner.versions)
        except FinalEvaluationError:
            raise
        except Exception:
            raise FinalEvaluationError("Final evaluation failed: inconsistent fitted model, final data or metrics") from None
