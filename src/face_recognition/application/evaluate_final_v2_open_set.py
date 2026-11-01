"""Independent final V2 evaluation. Operational decisions NEVER use holdout-derived thresholds."""

from dataclasses import fields
from math import isfinite

import numpy as np

from face_recognition.application.final_evaluation_data import FinalEvaluationError
from face_recognition.application.final_v2_holdout import FinalV2Data
from face_recognition.application.rejection_metrics import evaluate_threshold
from face_recognition.application.rejection_score_analysis import score_separation
from face_recognition.application.temporal_windows import aggregate_windows, build_windows
from face_recognition.domain.models.dataset import CONDITIONS
from face_recognition.domain.models.final_evaluation import ComparisonValues, FinalConditionMetrics, FinalPartitionCounts
from face_recognition.domain.models.final_v2_evaluation import FinalV2EvaluationReport
from face_recognition.domain.models.v2_open_set_config import FrozenV2OpenSetConfiguration


class FinalV2OpenSetEvaluator:
    def __init__(self, runner):
        self.runner = runner

    def evaluate(self, data, context):
        try:
            FinalV2Data(data.known_X, data.known_y, data.known_samples, data.unknown_X, data.unknown_samples,
                        data.class_ids, data.manifest, data.development_source)
            config = context.configuration
            if (not isinstance(config, FrozenV2OpenSetConfiguration)
                    or config != FrozenV2OpenSetConfiguration(config.provenance, config.development_evidence, config.known_cross_session_folds)
                    or type(context.threshold) not in (int, float) or not isfinite(context.threshold)
                    or self.runner.frozen_threshold != context.threshold or self.runner.class_ids != data.class_ids
                    or context.frozen_config_sha256 != data.manifest.frozen_config_sha256
                    or context.development_report_sha256 != data.manifest.development_report_sha256
                    or context.development_identity_split_sha256 != data.manifest.development_identity_split_sha256
                    or context.development_unknown_split_sha256 != data.manifest.development_unknown_split_sha256
                    or dict(context.model_fingerprints).get("manifest.json") != data.manifest.model_manifest_sha256
                    or context.processing_manifest_sha256 != data.manifest.processing_manifest_sha256):
                raise ValueError
            a = config.architecture
            kp, up = build_windows(data.known_samples, a.temporal_window_size), build_windows(data.unknown_samples, a.temporal_window_size)
            if not kp.windows or not up.windows:
                raise FinalEvaluationError("Independent final V2 populations require complete window groups")
            output = self.runner.infer(data.known_X, data.known_y, data.unknown_X)
            if any(tuple(name for name, _ in f.confidence) != (a.rejection_score_family,) for f in (output.known, output.unknown)):
                raise ValueError
            kw = aggregate_windows(kp, output.known.predicted, output.known.decisions, output.known.confidence)
            uw = aggregate_windows(up, output.unknown.predicted, output.unknown.decisions, output.unknown.confidence)
            labels = {p: i for i, p in enumerate(data.class_ids)}
            correct = kw.predicted == np.array([labels[w.person_id] for w in kp.windows], np.int32)
            ks, us = kw.confidence[0][1], uw.confidence[0][1]
            metrics = evaluate_threshold(ks, correct, us, context.threshold, allow_negative=True)
            # Diagnostic EER enumerates score cutpoints, but NEVER calibrates/changes the operational threshold.
            frames = score_separation(output.known.confidence[0][1], output.unknown.confidence[0][1])
            temporal = score_separation(ks, us)
            conditions = []
            for condition in CONDITIONS:
                kframes = np.array([s.condition == condition for s in data.known_samples])
                uframes = np.array([s.condition == condition for s in data.unknown_samples])
                kmask, umask = np.asarray(kw.conditions) == condition, np.asarray(uw.conditions) == condition
                nk, nu = int(kmask.sum()), int(umask.sum())
                if not kframes.any() and not uframes.any():
                    continue
                conditions.append(FinalConditionMetrics(condition, int(kframes.sum()), int(uframes.sum()), nk, nu,
                    float((output.known.predicted[kframes] == data.known_y[kframes]).mean()) if kframes.any() else None,
                    float((ks[kmask] >= context.threshold).mean()) if nk else None,
                    float((ks[kmask] < context.threshold).mean()) if nk else None,
                    float(((ks[kmask] >= context.threshold) & correct[kmask]).mean()) if nk else None,
                    float((us[umask] < context.threshold).mean()) if nu else None,
                    float((us[umask] >= context.threshold).mean()) if nu else None))
            evidence = config.development_evidence
            development = ComparisonValues(evidence.mean_roc_auc, evidence.mean_eer, evidence.metrics.unknown_false_acceptance_rate,
                                            evidence.metrics.known_rejection_rate, evidence.metrics.known_correct_identification_rate,
                                            evidence.metrics.open_set_exact_accuracy)
            final = ComparisonValues(temporal.roc_auc, temporal.equal_error_point.eer, metrics.unknown_false_acceptance_rate,
                                      metrics.known_rejection_rate, metrics.known_correct_identification_rate, metrics.open_set_exact_accuracy)
            delta = ComparisonValues(*(getattr(final, f.name) - getattr(development, f.name) for f in fields(ComparisonValues)))
            def counts(rows):
                return FinalPartitionCounts(len({s.person_id for s in rows}), len({(s.person_id, s.session_id) for s in rows}),
                                             len({s.source_relative_path for s in rows}), len(rows))
            return FinalV2EvaluationReport(context, counts(data.known_samples), counts(data.unknown_samples), output.closed_set,
                    frames, kp.counts, up.counts, metrics, temporal, tuple(conditions), kw.vote_stability, development, final, delta, self.runner.versions,
                    tuple((f.name, getattr(output.closed_set.metrics, f.name) - getattr(evidence.closed_set_frame_metrics, f.name))
                          for f in fields(output.closed_set.metrics)))
        except FinalEvaluationError:
            raise
        except Exception:
            raise FinalEvaluationError("Final V2 evaluation failed: inconsistent immutable model, independent data or metrics") from None
