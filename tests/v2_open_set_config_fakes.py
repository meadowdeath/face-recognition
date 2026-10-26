"""Synthetic aggregate reports only: no features, model fitting or dataset files."""

from dataclasses import asdict
import json

import numpy as np

from face_recognition.application.v2_rejection import binary_matrix, configurations, confusion, matrix_metrics, rank_key
from face_recognition.domain.models.baseline_evaluation import ClassificationMetrics
from face_recognition.domain.models.final_evaluation import FinalClosedSet, IdentityClassMetrics
from face_recognition.domain.models.v2_diagnostics import StudyFold, V1Reference, V2DiagnosticReport


def synthetic_v2_report():
    ids = ("synthetic_a", "synthetic_b")
    folds = (StudyFold("Fold A", ("visit_a", "visit_b"), "visit_c"),
             StudyFold("Fold B", ("visit_a", "visit_c"), "visit_b"),
             StudyFold("Fold C", ("visit_b", "visit_c"), "visit_a"))
    perfect = asdict(matrix_metrics(np.eye(3, dtype=int)))
    candidates = []
    for cfg in configurations():
        complete = 14 // cfg.window
        n = complete * 2
        counts = {"source_samples": 28, "used_samples": n * cfg.window,
                  "discarded_samples": 28 - n * cfg.window, "windows": n}
        matrix = np.diag([3 * n, 3 * n, 3 * n]).astype(int)
        metrics = asdict(matrix_metrics(matrix))
        auc = .9 if cfg.score_family == "svm_top1" and cfg.window == 7 else .6
        eer = 1 - auc
        cells = []
        for fold in folds:
            for rotation in (1, 2):
                cells.append({"fold": fold.name, "held_unknown_rotation": rotation,
                              "known_calibration_windows": n, "unknown_calibration_windows": complete,
                              "unknown_evaluation_windows": complete, "global_threshold": .5,
                              "global_calibration_metrics": perfect,
                              "held_unknown_evaluation_far": 0.,
                              "known_calibration_and_held_unknown_evaluation_metrics": perfect,
                              "roc_auc": auc, "eer": {"eer": eer, "threshold": .5, "far": eer, "frr": eer},
                              "identity_thresholds": [{"person_id": person, "known_support": complete, "unknown_support": 0,
                                                        "eligible_known_support": complete, "eligible_unknown_support": 0,
                                                        "threshold": None, "used_threshold": .5, "available": False} for person in ids] if cfg.per_identity else [],
                              "known_fallback_observations": n if cfg.per_identity else 0,
                              "unknown_evaluation_fallback_observations": complete if cfg.per_identity else 0,
                              "far_constrained_calibration_points": [] if cfg.per_identity else [
                                  {"far_target": target, "status": "available", "threshold": .5,
                                   "calibration_metrics": perfect, "held_unknown_evaluation_metrics": perfect} for target in (.2, .1, .05)]})
        votes = {"unanimous_windows": n, "majority_not_unanimous_windows": 0, "tied_windows": 0,
                 "rates": {"unanimous_rate": 1., "majority_not_unanimous_rate": 0., "tie_rate": 0.}}
        candidates.append({"identifier": cfg.identifier, "configuration": asdict(cfg), "cells": cells,
                           "window_counts": [{"fold": f.name, "known_unique_within_fold": counts,
                                              "unknown_reobserved_under_each_known_model": counts} for f in folds],
                           "votes": [{"fold": f.name, "known": votes, "unknown": votes,
                                      "known_mean_agreement": 1., "unknown_mean_agreement": 1.} for f in folds],
                           "metrics": metrics, "open_set_confusion": asdict(confusion(matrix, ids + ("UNKNOWN",), 2)),
                           "binary_confusion": asdict(confusion(binary_matrix(matrix), ("KNOWN", "UNKNOWN"), 2)),
                           "mean_roc_auc": auc, "mean_eer": eer,
                           "roc_auc_statistics": {"mean": auc, "population_std": 0., "minimum": auc, "maximum": auc},
                           "eer_statistics": {"mean": eer, "population_std": 0., "minimum": eer, "maximum": eer},
                           "global_threshold_statistics": {"mean": .5, "population_std": 0., "minimum": .5, "maximum": .5},
                           "per_identity_threshold_statistics": [{"person_id": person, "available_cells": 0, "total_cells": 6,
                                                                  "available_threshold_statistics": None} for person in ids] if cfg.per_identity else [],
                           "known_identity_diagnostics": [{"person_id": person, "source_samples": 42, "closed_set_accuracy": 1.,
                                                            "unique_windows": 3 * complete, "window_observations": 6 * complete,
                                                            "known_acceptance_rate": 1., "known_rejection_rate": 0.,
                                                            "known_correct_identification_rate": 1., "known_misidentification_rate": 0.} for person in ids],
                           "known_condition_diagnostics": [{"condition": "center", "source_samples": 84,
                                                             "closed_set_accuracy": 1., "unique_windows": 3 * n,
                                                             "window_observations": 6 * n, "known_acceptance_rate": 1.,
                                                             "known_rejection_rate": 0., "known_correct_identification_rate": 1., "known_misidentification_rate": 0.}],
                           "unknown_condition_diagnostics": [{"condition": "center", "unique_source_samples": 28,
                                                               "window_observations": 3 * n, "unknown_rejection_rate": 1., "unknown_false_acceptance_rate": 0.}],
                           "comparison_with_v1_final_descriptive_only": {"roc_auc_delta": auc - .7, "eer_delta": eer - .3,
                                                                         "far_delta": -.2, "frr_delta": -.4,
                                                                         "known_correct_identification_delta": .5, "open_set_accuracy_delta": .4}})
    distributions = []
    for fold in ("Fold A", "Fold B", "Fold C", "pooled_known_OOF_unknown_three_model_observations"):
        for score in ("svm_top1", "svm_margin", "mean_5nn", "centroid_distance"):
            n = 28 if fold.startswith("Fold") else 84
            def stats(mean):
                return {"count": n, "statistics": {"count": n, "mean": mean, "std": .2, "median": mean,
                                                    "p10": mean - .2, "p25": mean - .1, "p75": mean + .1, "p90": mean + .2}}
            distributions.append({"fold": fold, "score_family": score, "known_correct": stats(.6),
                                  "known_incorrect": {"count": 0, "statistics": None}, "unknown": stats(.4)})
            if fold.startswith("Fold"):
                distributions[-1].update(roc_auc=.6, eer=.4, unknown_at_or_above_known_median=.2, known_below_unknown_median=.2)
    closed = FinalClosedSet(84, ClassificationMetrics(1., 1., 1., 1.), ids, ((42, 0), (0, 42)),
                           tuple(IdentityClassMetrics(person, 1., 1., 1., 42) for person in ids), scope="V2 out-of-session known frames; DEVELOPMENT ONLY")
    ranking = tuple(r["identifier"] for r in sorted(candidates, key=rank_key))
    report = V2DiagnosticReport(V1Reference("a" * 64, "b" * 64, .8, .8, .7, .3, .2, .4, .5, .6),
                                ids, 2, 84, 28, folds, closed, tuple(distributions), tuple(candidates), ranking, ranking[0])
    return json.loads(json.dumps(asdict(report)))
