"""V2 development study. Never mutates or refits any persisted V1 artifact.

Known thresholds are calibrated/described out of session, not independently
validated. Unknown identities receive leave-one-identity-out threshold evaluation.
The old holdouts are explicitly promoted to development, with no final V2 claim.
"""

from collections import defaultdict
from dataclasses import asdict

import numpy as np

from face_recognition.application.rejection_score_analysis import score_separation
from face_recognition.application.temporal_windows import aggregate_windows, build_windows
from face_recognition.application.v2_rejection import (
    binary_matrix, calibrate, configurations, confusion, distribution, far_points,
    identity_thresholds, matrix_metrics, open_matrix, rank_key, rotation_masks,
)
from face_recognition.application.v2_study_data import V2StudyError, study_folds
from face_recognition.domain.models.v2_diagnostics import V2DiagnosticReport, V2_SCORES


def _moments(values):
    return {"mean": float(np.mean(values)), "population_std": float(np.std(values)),
            "minimum": float(np.min(values)), "maximum": float(np.max(values))}


def _windows(samples, frames, size):
    plan = build_windows(samples, size)
    observations = aggregate_windows(plan, frames.predicted, frames.decisions, frames.confidence)
    if not plan.windows:
        raise V2StudyError("Every evaluated role requires complete temporal windows")
    agreement = np.array([np.count_nonzero(frames.predicted[list(w.indices)] == p) / size
                          for w, p in zip(plan.windows, observations.predicted)])
    return plan, observations, agreement


def _known_diagnostic(counter):
    count, accepted, correct = map(int, counter)
    return {"window_observations": count, "known_acceptance_rate": accepted / count if count else None,
            "known_rejection_rate": (count - accepted) / count if count else None,
            "known_correct_identification_rate": correct / count if count else None,
            "known_misidentification_rate": (accepted - correct) / count if count else None}


class V2DiagnosticEvaluator:
    """Three sequential fits, then six rules and seven temporal policies per rule.

    The runner owns sklearn details; all threshold fitting remains visibly inside
    each known-session / held-unknown-identity rotation.
    """

    def evaluate(self, data, baseline, runner):
        ids = data.source.specification.known_ids
        folds, configs = study_folds(data.source), configurations()
        identities, rotations = len(ids), len(data.source.specification.unknown_ids)
        labels = {p: i for i, p in enumerate(ids)}
        states = {c.identifier: {"configuration": asdict(c), "identifier": c.identifier,
                                 "matrix": np.zeros((identities + 1, identities + 1), dtype=np.int64),
                                 "cells": [], "known_people": defaultdict(lambda: np.zeros(3, dtype=np.int64)),
                                 "known_conditions": defaultdict(lambda: np.zeros(3, dtype=np.int64)),
                                 "unknown_conditions": defaultdict(lambda: np.zeros(2, dtype=np.int64)),
                                 "window_counts": [], "votes": [], "global_thresholds": []} for c in configs}
        oof = np.full(len(data.known_y), -1, dtype=np.int32)
        distributions = []
        pooled = {score: [[], [], []] for score in V2_SCORES}
        for fold in folds:
            held = np.array([s.session_id == fold.calibration_session for s in data.known_samples])
            if not held.any() or held.all():
                raise V2StudyError("Invalid known-session development fold")
            samples = tuple(s for s, selected in zip(data.known_samples, held) if selected)
            truth_frames = data.known_y[held]
            known, unknown = runner.infer_fold(data.known_X[~held], data.known_y[~held], data.known_X[held],
                                               truth_frames, data.unknown_X, identities)
            oof[held] = known.predicted
            for score in V2_SCORES:
                ks, us = dict(known.confidence)[score], dict(unknown.confidence)[score]
                correct = known.predicted == truth_frames
                for group, values in zip(pooled[score], (ks[correct], ks[~correct], us)):
                    group.append(values)
                separation = score_separation(ks, us)
                distributions.append({"fold": fold.name, "score_family": score,
                                      "known_correct": asdict(distribution(ks[correct])),
                                      "known_incorrect": asdict(distribution(ks[~correct])),
                                      "unknown": asdict(distribution(us)), "roc_auc": separation.roc_auc,
                                      "eer": separation.equal_error_point.eer,
                                      "unknown_at_or_above_known_median": float(np.mean(us >= np.median(ks))),
                                      "known_below_unknown_median": float(np.mean(ks < np.median(us)))})
            for size in (1, 3, 5, 7):
                kp, ko, ka = _windows(samples, known, size)
                up, uo, ua = _windows(data.unknown_samples, unknown, size)
                truth = np.array([labels[w.person_id] for w in kp.windows], dtype=np.int32)
                unknown_people = tuple(w.person_id for w in up.windows)
                if len(set(unknown_people)) != rotations:
                    raise V2StudyError("Every unknown identity must have complete windows at every size")
                correct = ko.predicted == truth
                for cfg in (c for c in configs if c.window == size):
                    state = states[cfg.identifier]
                    ks, us = dict(ko.confidence)[cfg.score_family], dict(uo.confidence)[cfg.score_family]
                    kg = ka * size >= size // 2 + 1 if cfg.strict_majority_gate else np.ones(len(ka), dtype=bool)
                    ug = ua * size >= size // 2 + 1 if cfg.strict_majority_gate else np.ones(len(ua), dtype=bool)
                    state["window_counts"].append({"fold": fold.name, "known_unique_within_fold": asdict(kp.counts),
                                                   "unknown_reobserved_under_each_known_model": asdict(up.counts)})
                    state["votes"].append({"fold": fold.name, "known": asdict(ko.vote_stability),
                                           "unknown": asdict(uo.vote_stability), "known_mean_agreement": float(ka.mean()),
                                           "unknown_mean_agreement": float(ua.mean())})
                    for rotation, (cal, test) in enumerate(rotation_masks(unknown_people), 1):
                        threshold, _, _, calibration_metrics = calibrate(ks, correct, us[cal], kg, ug[cal])
                        state["global_thresholds"].append(threshold)
                        per_id = identity_thresholds(ids, ks, correct, us[cal], ko.predicted, uo.predicted[cal], kg,
                                                     ug[cal], threshold) if cfg.per_identity else ()
                        thresholds = np.array([r.used_threshold for r in per_id]) if per_id else np.full(identities, threshold)
                        accepted, unknown_accepted = kg & (ks >= thresholds[ko.predicted]), ug[test] & (us[test] >= thresholds[uo.predicted[test]])
                        matrix = open_matrix(truth, ko.predicted, accepted, uo.predicted[test], unknown_accepted, identities)
                        state["matrix"] += matrix
                        separation = score_separation(ks, us[test])
                        cell_metrics = matrix_metrics(matrix)
                        points = far_points(ks, correct, us[cal], kg, ug[cal]) if not cfg.per_identity else ()
                        for point in points:
                            t = point["threshold"]
                            diagnostic_matrix = open_matrix(truth, ko.predicted, kg & (ks >= t),
                                                            uo.predicted[test], ug[test] & (us[test] >= t), identities)
                            point["held_unknown_evaluation_metrics"] = asdict(matrix_metrics(diagnostic_matrix))
                        state["cells"].append({"fold": fold.name, "held_unknown_rotation": rotation,
                                                "known_calibration_windows": len(ks), "unknown_calibration_windows": int(cal.sum()),
                                                "unknown_evaluation_windows": int(test.sum()), "global_threshold": threshold,
                                                "global_calibration_metrics": asdict(calibration_metrics),
                                                "held_unknown_evaluation_far": cell_metrics.unknown_false_acceptance_rate,
                                                "known_calibration_and_held_unknown_evaluation_metrics": asdict(cell_metrics),
                                                "roc_auc": separation.roc_auc, "eer": asdict(separation.equal_error_point),
                                                "identity_thresholds": tuple(asdict(r) for r in per_id),
                                                "known_fallback_observations": int(sum(np.count_nonzero(ko.predicted == i) for i, r in enumerate(per_id) if not r.available)),
                                                "unknown_evaluation_fallback_observations": int(sum(np.count_nonzero(uo.predicted[test] == i) for i, r in enumerate(per_id) if not r.available)),
                                                "far_constrained_calibration_points": points})
                        for w, pred, true, accept in zip(kp.windows, ko.predicted, truth, accepted):
                            values = (1, int(accept), int(accept and pred == true))
                            state["known_people"][w.person_id] += values
                            state["known_conditions"][w.condition] += values
                        for w, accept in zip((w for w, selected in zip(up.windows, test) if selected), unknown_accepted):
                            state["unknown_conditions"][w.condition] += (1, int(not accept))
        if np.any(oof < 0):
            raise V2StudyError("Every known source sample must have exactly one out-of-session prediction")
        closed = runner.summarize_identity(data.known_y, oof, ids)
        class_metrics = {r.person_id: asdict(r) for r in closed.per_class}
        candidates = []
        for cfg in configs:
            state = states[cfg.identifier]
            # Preserve source support even when an entire group has no complete
            # window: unavailable rates are null, not an omitted condition/person.
            for person in ids:
                state["known_people"][person]
            for sample in data.known_samples:
                state["known_conditions"][sample.condition]
            for sample in data.unknown_samples:
                state["unknown_conditions"][sample.condition]
            matrix = state.pop("matrix")
            adjusted = matrix.astype(np.float64)
            adjusted[:-1] /= rotations
            metrics = asdict(matrix_metrics(adjusted))
            state["metrics"] = metrics
            state["open_set_confusion"] = asdict(confusion(matrix, ids + ("UNKNOWN",), rotations))
            state["binary_confusion"] = asdict(confusion(binary_matrix(matrix), ("KNOWN", "UNKNOWN"), rotations))
            state["mean_roc_auc"] = float(np.mean([c["roc_auc"] for c in state["cells"]]))
            state["mean_eer"] = float(np.mean([c["eer"]["eer"] for c in state["cells"]]))
            state["roc_auc_statistics"] = _moments([c["roc_auc"] for c in state["cells"]])
            state["eer_statistics"] = _moments([c["eer"]["eer"] for c in state["cells"]])
            state["global_threshold_statistics"] = _moments(state.pop("global_thresholds"))
            state["per_identity_threshold_statistics"] = tuple(
                {"person_id": person, "available_cells": len(values), "total_cells": len(state["cells"]),
                 "available_threshold_statistics": _moments(values) if values else None}
                for person in ids
                for values in [[r["threshold"] for c in state["cells"] for r in c["identity_thresholds"]
                                if r["person_id"] == person and r["available"]]]) if cfg.per_identity else ()
            state["known_identity_diagnostics"] = tuple(
                {"person_id": person, "source_samples": sum(s.person_id == person for s in data.known_samples),
                 "closed_set_accuracy": class_metrics[person]["recall"], "closed_set_class_metrics": class_metrics[person],
                 "unique_windows": int(counter[0]) // rotations, **_known_diagnostic(counter)}
                for person, counter in sorted(state.pop("known_people").items()))
            state["known_condition_diagnostics"] = tuple(
                {"condition": condition, "source_samples": int(sum(s.condition == condition for s in data.known_samples)),
                 "closed_set_accuracy": float(np.mean(oof[mask] == data.known_y[mask])),
                 "unique_windows": int(counter[0]) // rotations, **_known_diagnostic(counter)}
                for condition, counter in sorted(state.pop("known_conditions").items())
                for mask in [np.array([s.condition == condition for s in data.known_samples])])
            state["unknown_condition_diagnostics"] = tuple(
                {"condition": condition, "unique_source_samples": sum(s.condition == condition for s in data.unknown_samples),
                 "window_observations": int(counter[0]), "unknown_rejection_rate": int(counter[1]) / int(counter[0]) if counter[0] else None,
                 "unknown_false_acceptance_rate": 1 - int(counter[1]) / int(counter[0]) if counter[0] else None}
                for condition, counter in sorted(state.pop("unknown_conditions").items()))
            state["comparison_with_v1_final_descriptive_only"] = {
                "roc_auc_delta": state["mean_roc_auc"] - baseline.roc_auc, "eer_delta": state["mean_eer"] - baseline.eer,
                "far_delta": metrics["unknown_false_acceptance_rate"] - baseline.far,
                "frr_delta": metrics["known_rejection_rate"] - baseline.frr,
                "known_correct_identification_delta": metrics["known_correct_identification_rate"] - baseline.known_correct_identification,
                "open_set_accuracy_delta": metrics["open_set_exact_accuracy"] - baseline.open_set_accuracy}
            candidates.append(state)
        for score, groups in pooled.items():
            arrays = [np.concatenate(group) for group in groups]
            distributions.append({"fold": "pooled_known_OOF_unknown_three_model_observations", "score_family": score,
                                  **{name: asdict(distribution(values)) for name, values in zip(("known_correct", "known_incorrect", "unknown"), arrays)}})
        ranking = tuple(r["identifier"] for r in sorted(candidates, key=rank_key))
        return V2DiagnosticReport(baseline, ids, rotations, len(data.known_samples), len(data.unknown_samples), folds,
                                  closed, tuple(distributions), tuple(candidates), ranking, ranking[0])
