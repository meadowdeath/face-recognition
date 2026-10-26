"""Three OOF calibration fits, fixed global threshold, then ONE all-session refit."""

from dataclasses import dataclass, field, fields
from math import isfinite
from statistics import fmean, median, pstdev

import numpy as np

from face_recognition.application.rejection_score_analysis import analyze_scores, score_distribution
from face_recognition.application.temporal_windows import aggregate_windows, build_windows
from face_recognition.application.v2_study_data import V2StudyData, study_folds
from face_recognition.domain.models.open_set_model import OpenSetModelError
from face_recognition.domain.models.temporal_open_set import WindowCounts
from face_recognition.domain.models.v2_open_set_config import FrozenV2OpenSetConfiguration
from face_recognition.domain.models.v2_open_set_model import (
    FoldThresholdStability, V2CalibrationFold, V2RefitSummary, V2TransferGuard, V2TransferValidation,
)


def threshold_stability(values):
    values = tuple(float(v) for v in values)
    if len(values) != 3 or not all(isfinite(v) for v in values):
        raise OpenSetModelError("Exactly three finite fold thresholds are required")
    return FoldThresholdStability(values, min(values), max(values), max(values) - min(values),
                                  fmean(values), median(values), pstdev(values))


def calibration_scale(known, unknown, stability):
    if not isinstance(stability, FoldThresholdStability) or stability != threshold_stability(stability.thresholds):
        raise OpenSetModelError("Invalid V2 fold-threshold stability diagnostics")
    pooled = score_distribution(np.concatenate((known, unknown)))
    guard = V2TransferGuard()
    unit = max(pooled.p90 - pooled.p10, guard.minimum_score_unit)
    if not isfinite(unit) or unit <= 0 or not all(isfinite(v) for v in vars(pooled).values()):
        raise OpenSetModelError("Non-finite V2 calibration scale diagnostics")
    if stability.range / unit > guard.failure_units:
        raise OpenSetModelError("Frozen V2 direct-transfer assumption failed: inconsistent fold threshold scales; a new DEVELOPMENT protocol/version is required")
    return unit


def validate_direct_transfer(known_oof, unknown_oof, known_final, unknown_final, stability):
    unit = calibration_scale(known_oof, unknown_oof, stability)
    known, unknown, fk, fu = map(score_distribution, (known_oof, unknown_oof, known_final, unknown_final))
    if any(not all(isfinite(v) for v in vars(d).values()) for d in (known, unknown, fk, fu)):
        raise OpenSetModelError("Non-finite V2 direct-transfer diagnostics")
    shifts = (fk.mean - known.mean, fk.median - known.median, fu.mean - unknown.mean, fu.median - unknown.median)
    maximum = max(abs(v) / unit for v in shifts)
    guard = V2TransferGuard()
    if not isfinite(maximum) or maximum > guard.failure_units:
        raise OpenSetModelError("Frozen V2 direct-transfer assumption failed: final/development score scale mismatch; no threshold adjustment permitted; a new DEVELOPMENT protocol/version is required")
    return V2TransferValidation("passed", guard, unit, stability.range / unit, *shifts, maximum,
                                maximum > guard.warning_units or stability.range / unit > guard.warning_units,
                                known, unknown, fk, fu)


def _window_scores(plan, frames):
    if tuple(name for name, _ in frames.confidence) != ("svm_top1",):
        raise OpenSetModelError("Frozen V2 requires only svm_top1 confidence")
    return aggregate_windows(plan, frames.predicted, frames.decisions, frames.confidence)


def _sum_counts(counts):
    counts = tuple(counts)
    return WindowCounts(*(sum(getattr(c, f.name) for c in counts) for f in fields(WindowCounts)))


@dataclass(frozen=True, eq=False)
class FittedV2DevelopmentModel:
    summary: V2RefitSummary
    parts: object = field(repr=False)


class V2OpenSetRefitter:
    def __init__(self, runner):
        self.runner = runner

    def run(self, config, config_sha256, data):
        context = "Invalid frozen V2 configuration/development data"
        try:
            if (not isinstance(data, V2StudyData) or not isinstance(config, FrozenV2OpenSetConfiguration)
                    or config != FrozenV2OpenSetConfiguration(config.provenance, config.development_evidence, config.known_cross_session_folds)
                    or not isinstance(config_sha256, str) or len(config_sha256) != 64
                    or any(c not in "0123456789abcdef" for c in config_sha256)
                    or data.partition.source_identity_split_sha256 != config.provenance.original_identity_split_sha256
                    or study_folds(data.source) != config.known_cross_session_folds):
                raise ValueError
            e, a = config.development_evidence, config.architecture
            if ((len(data.known_samples), len(data.unknown_samples), len(data.source.specification.known_ids),
                 len(data.source.specification.unknown_ids)) != (e.known_source_samples, e.unknown_source_samples,
                                                                  e.known_identity_count, e.unknown_identity_count)):
                raise ValueError
            ids = data.source.specification.known_ids
            labels = {person: i for i, person in enumerate(ids)}
            up = build_windows(data.unknown_samples, a.temporal_window_size)
            kp_all = build_windows(data.known_samples, a.temporal_window_size)
            if (not up.windows or not kp_all.windows or len(kp_all.windows) != e.known_unique_windows
                    or 3 * len(up.windows) != e.unknown_three_model_window_observations):
                raise ValueError
            known_scores, unknown_scores, correct, results = [], [], [], []
            context = "V2 cross-session fit/scoring failed; only known fit sessions may fit preprocessing/SVM"
            for fold in config.known_cross_session_folds:
                held = np.array([s.session_id == fold.calibration_session for s in data.known_samples])
                samples = tuple(s for s, yes in zip(data.known_samples, held) if yes)
                kp = build_windows(samples, a.temporal_window_size)
                if not kp.windows or held.all() or not held.any():
                    raise ValueError
                # Three ephemeral models; UNKNOWN and held known rows only score.
                model = self.runner.fit_model(config, data.known_X[~held], data.known_y[~held], ids)
                kw = _window_scores(kp, self.runner.infer(model, data.known_X[held]))
                uw = _window_scores(up, self.runner.infer(model, data.unknown_X))
                del model
                k, u = kw.confidence[0][1], uw.confidence[0][1]
                c = kw.predicted == np.array([labels[w.person_id] for w in kp.windows], np.int32)
                known_scores.append(k)
                unknown_scores.append(u)
                correct.append(c)
                results.append(V2CalibrationFold(fold, int((~held).sum()), kp.counts, up.counts, analyze_scores(k, c, u)))
            K, U, C = np.concatenate(known_scores), np.concatenate(unknown_scores), np.concatenate(correct)
            context = "V2 pooled OOF window threshold calibration failed"
            analysis = analyze_scores(K, C, U)  # Operational threshold chosen exactly ONCE from pooled OOF scores.
            stability = threshold_stability(r.analysis.calibration.threshold for r in results)
            calibration_scale(K, U, stability)  # Reject gross fold mismatch before the final all-session fit.
            context = "Final V2 all-session known-only refit failed"
            parts = self.runner.fit_model(config, data.known_X, data.known_y, ids)
            # These resubstitution scores are diagnostic ONLY, never passed to analyze_scores.
            final_known = _window_scores(kp_all, self.runner.infer(parts, data.known_X)).confidence[0][1]
            final_unknown = _window_scores(up, self.runner.infer(parts, data.unknown_X)).confidence[0][1]
            transfer = validate_direct_transfer(K, U, final_known, final_unknown, stability)
            summary = V2RefitSummary(config_sha256, data.partition.source_identity_split_sha256,
                                     len(data.known_samples), len(data.unknown_samples), len(ids), a.original_feature_count,
                                     a.selected_features, self.runner.support_count(parts),
                                     tuple(sorted({s.session_id for s in data.known_samples})), tuple(results),
                                     _sum_counts(r.known_counts for r in results), up.counts,
                                     _sum_counts(r.unknown_counts for r in results), analysis, stability, transfer, self.runner.versions)
            return FittedV2DevelopmentModel(summary, parts)
        except OpenSetModelError:
            raise
        except Exception:
            raise OpenSetModelError(context) from None
