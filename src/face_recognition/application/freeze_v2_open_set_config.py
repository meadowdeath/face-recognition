"""Validate aggregate V2 evidence and freeze policy only. No model/data loader."""

from dataclasses import MISSING, asdict, fields
from math import isfinite
from statistics import fmean, pstdev

import numpy as np

from face_recognition.application.freeze_open_set_config import _close, _counts, _finite_tree, _metrics, _object, _rate, _require
from face_recognition.application.v2_rejection import binary_matrix, configurations, matrix_metrics, rank_key
from face_recognition.domain.models.baseline_evaluation import ClassificationMetrics
from face_recognition.domain.models.dataset import validate_identifier
from face_recognition.domain.models.open_set_config import ConfigurationFreezeError
from face_recognition.domain.models.rejection_score_evaluation import ScoreDistribution
from face_recognition.domain.models.temporal_open_set import VoteStability
from face_recognition.domain.models.v2_diagnostics import StudyFold, V1Reference, V2DiagnosticReport, V2_SCORES
from face_recognition.domain.models.v2_open_set_config import (
    FrozenV2OpenSetConfiguration, V2DevelopmentEvidence, V2ReportProvenance, V2_FROZEN_CANDIDATE,
)


def _integer(value, *, minimum=1):
    _require(type(value) is int and value >= minimum, "invalid aggregate count")
    return value


def _number(value):
    _require(type(value) in (int, float) and isfinite(value), "invalid finite numeric metadata")
    return float(value)


def _matrix(value, n):
    _require(isinstance(value, list) and len(value) == n
             and all(isinstance(row, list) and len(row) == n for row in value), "invalid confusion dimensions")
    _require(all(type(v) is int and v >= 0 for row in value for v in row), "invalid confusion counts")
    return np.asarray(value, dtype=np.int64)


def _confusion(value, labels, rotations):
    _require(isinstance(value, dict) and value["labels"] == list(labels), "inconsistent confusion labels")
    matrix = _matrix(value["observation_counts"], len(labels))
    adjusted = matrix.astype(np.float64)
    adjusted[:-1] /= rotations
    normalized = np.divide(matrix, matrix.sum(axis=1, keepdims=True), out=np.zeros(matrix.shape),
                           where=matrix.sum(axis=1, keepdims=True) != 0)
    _require(np.shape(value["role_adjusted_counts"]) == matrix.shape
             and np.allclose(value["role_adjusted_counts"], adjusted, rtol=1e-12, atol=1e-12)
             and np.shape(value["row_normalized_rates"]) == matrix.shape
             and np.allclose(value["row_normalized_rates"], normalized, rtol=1e-12, atol=1e-12), "inconsistent confusion rates/count adjustment")
    return matrix, adjusted


def _moments(actual, values):
    expected = {"mean": fmean(values), "population_std": pstdev(values), "minimum": min(values), "maximum": max(values)}
    _require(isinstance(actual, dict) and set(actual) == set(expected)
             and all(_close(_number(actual[k]), v) for k, v in expected.items()), "summary statistics disagree with fold evidence")


def _closed(report, ids):
    closed = report["closed_set_frame_oof"]
    _require(closed["label_order"] == ids and closed["samples"] == report["known_source_samples"], "incomplete known OOF evidence")
    matrix = _matrix(closed["confusion_matrix"], len(ids))
    _require(int(matrix.sum()) == report["known_source_samples"], "OOF counts disagree with source")
    p, r, f = [], [], []
    _require(len(closed["per_class"]) == len(ids), "missing known class evidence")
    for i, (person, row) in enumerate(zip(ids, closed["per_class"])):
        tp, support, predicted = int(matrix[i, i]), int(matrix[i].sum()), int(matrix[:, i].sum())
        precision, recall = tp / predicted if predicted else 0., tp / support if support else 0.
        f1 = 2 * tp / (support + predicted) if support + predicted else 0.
        _require(row["person_id"] == person and row["support"] == support
                 and all(_close(_rate(row[k]), v) for k, v in (("precision", precision), ("recall", recall), ("f1", f1))),
                 "inconsistent closed-set class metrics")
        p.append(precision)
        r.append(recall)
        f.append(f1)
    expected = ClassificationMetrics(float(np.trace(matrix) / matrix.sum()), fmean(p), fmean(r), fmean(f))
    _require(all(_close(_rate(closed["metrics"][k]), v) for k, v in asdict(expected).items()), "inconsistent closed-set aggregate metrics")
    return expected


def _distributions(report, folds):
    rows = report["score_distributions"]
    pooled = "pooled_known_OOF_unknown_three_model_observations"
    expected = [(name, score) for name in [f.name for f in folds] + [pooled] for score in V2_SCORES]
    _require(isinstance(rows, list) and [(r["fold"], r["score_family"]) for r in rows] == expected,
             "incomplete score distributions")
    frame_counts = {w["fold"]: w["known_unique_within_fold"]["source_samples"] for w in report["candidates"][0]["window_counts"]}
    known_counts = {}
    for row in rows:
        counts = []
        for group in ("known_correct", "known_incorrect", "unknown"):
            entry = row[group]
            n = _integer(entry["count"], minimum=0)
            counts.append(n)
            if not n:
                _require(entry["statistics"] is None, "empty score group must have null statistics")
            else:
                stats = _object(entry["statistics"], ScoreDistribution)
                _require(stats["count"] == n and _number(stats["std"]) >= 0, "inconsistent distribution count/std")
                _number(stats["mean"])
                quantiles = [_number(stats[k]) for k in ("p10", "p25", "median", "p75", "p90")]
                _require(quantiles == sorted(quantiles), "inconsistent distribution percentiles")
        if row["fold"] == pooled:
            _require(counts[:2] == [sum(v[i] for v in known_counts.values()) for i in (0, 1)]
                     and counts[2] == 3 * report["unknown_source_samples"], "pooled score supports disagree with folds")
        else:
            _require(sum(counts[:2]) == frame_counts[row["fold"]] and counts[2] == report["unknown_source_samples"],
                     "score distribution/source supports disagree")
            prior = known_counts.setdefault(row["fold"], counts[:2])
            _require(prior == counts[:2], "SVM correctness supports differ across score families")
            for key in ("roc_auc", "eer", "unknown_at_or_above_known_median", "known_below_unknown_median"):
                _rate(row[key])
    correct = int(np.trace(np.asarray(report["closed_set_frame_oof"]["confusion_matrix"])))
    _require(sum(v[0] for v in known_counts.values()) == correct, "score correctness disagrees with OOF confusion")


def _votes(value, count):
    value = _object(value, VoteStability)
    values = [_integer(value[key], minimum=0) for key in ("unanimous_windows", "majority_not_unanimous_windows", "tied_windows")]
    _require(sum(values) == count, "inconsistent vote support")
    for key, n in zip(("unanimous_rate", "majority_not_unanimous_rate", "tie_rate"), values):
        _require(_close(_rate(value["rates"][key]), n / count), "inconsistent vote rates")


def _candidate(row, cfg, ids, folds, rotations, report):
    required = {"configuration", "identifier", "cells", "window_counts", "votes", "metrics", "open_set_confusion",
                "binary_confusion", "mean_roc_auc", "mean_eer", "roc_auc_statistics", "eer_statistics", "global_threshold_statistics",
                "per_identity_threshold_statistics", "known_identity_diagnostics", "known_condition_diagnostics",
                "unknown_condition_diagnostics", "comparison_with_v1_final_descriptive_only"}
    _require(isinstance(row, dict) and set(row) == required and row["identifier"] == cfg.identifier
             and row["configuration"] == asdict(cfg) and row["configuration"]["per_identity"] is cfg.per_identity
             and row["configuration"]["strict_majority_gate"] is cfg.strict_majority_gate
             and type(row["configuration"]["window"]) is int, "unexpected rejection strategy/window/gate")
    windows = row["window_counts"]
    _require(isinstance(windows, list) and len(windows) == 3 and [w["fold"] for w in windows] == [f.name for f in folds],
             "incomplete window fold evidence")
    counts = {w["fold"]: (_counts(w["known_unique_within_fold"], cfg.window),
                            _counts(w["unknown_reobserved_under_each_known_model"], cfg.window)) for w in windows}
    _require(sum(k.source_samples for k, _ in counts.values()) == report["known_source_samples"]
             and all(u.source_samples == report["unknown_source_samples"] for _, u in counts.values()), "window/source counts disagree")
    cells = row["cells"]
    _require(isinstance(cells, list) and len(cells) == 3 * rotations
             and [(c["fold"], c["held_unknown_rotation"]) for c in cells] == [(f.name, i) for f in folds for i in range(1, rotations + 1)],
             "incomplete UNKNOWN identity rotations")
    for cell in cells:
        known, unknown = counts[cell["fold"]]
        _require(cell["known_calibration_windows"] == known.windows
                 and _integer(cell["unknown_calibration_windows"]) + _integer(cell["unknown_evaluation_windows"]) == unknown.windows,
                 "inconsistent calibration/evaluation support")
        _number(cell["global_threshold"])
        _metrics(cell["global_calibration_metrics"])
        metrics = _metrics(cell["known_calibration_and_held_unknown_evaluation_metrics"])
        _require(_close(_rate(cell["held_unknown_evaluation_far"]), metrics.unknown_false_acceptance_rate), "inconsistent held-UNKNOWN FAR")
        _rate(cell["roc_auc"])
        eer = cell["eer"]
        _number(eer["threshold"])
        _require(_close(_rate(eer["eer"]), (_rate(eer["far"]) + _rate(eer["frr"])) / 2), "invalid EER evidence")
        if not cfg.per_identity:
            _require(cell["identity_thresholds"] == [] and len(cell["far_constrained_calibration_points"]) == 3,
                     "global threshold evidence is inconsistent")
            _require([p["far_target"] for p in cell["far_constrained_calibration_points"]] == [.2, .1, .05], "missing FAR diagnostics")
    for fold in folds:
        _require(sum(c["unknown_evaluation_windows"] for c in cells if c["fold"] == fold.name) == counts[fold.name][1].windows,
                 "UNKNOWN evaluation coverage is incomplete")
    auc, eer = [c["roc_auc"] for c in cells], [c["eer"]["eer"] for c in cells]
    _require(_close(_rate(row["mean_roc_auc"]), fmean(auc)) and _close(_rate(row["mean_eer"]), fmean(eer)), "candidate AUC/EER disagree with folds")
    _moments(row["roc_auc_statistics"], auc)
    _moments(row["eer_statistics"], eer)
    _moments(row["global_threshold_statistics"], [c["global_threshold"] for c in cells])
    matrix, adjusted = _confusion(row["open_set_confusion"], ids + ["UNKNOWN"], rotations)
    binary, _ = _confusion(row["binary_confusion"], ("KNOWN", "UNKNOWN"), rotations)
    _require(np.array_equal(binary, binary_matrix(matrix)) and int(matrix[:-1].sum()) == rotations * sum(k.windows for k, _ in counts.values())
             and int(matrix[-1].sum()) == sum(u.windows for _, u in counts.values()), "confusion/window support disagrees")
    actual, expected = _metrics(row["metrics"]), matrix_metrics(adjusted)
    _require(all(_close(asdict(actual)[k], v) for k, v in asdict(expected).items()), "operating metrics disagree with confusion evidence")
    _require(len(row["votes"]) == 3 and len(row["known_identity_diagnostics"]) == len(ids)
             and bool(row["known_condition_diagnostics"]) and bool(row["unknown_condition_diagnostics"]), "missing aggregate diagnostics")
    for vote, fold in zip(row["votes"], folds):
        _require(vote["fold"] == fold.name, "inconsistent vote fold")
        for role, count, key in (("known", counts[fold.name][0].windows, "known_mean_agreement"),
                                 ("unknown", counts[fold.name][1].windows, "unknown_mean_agreement")):
            _votes(vote[role], count)
            _rate(vote[key])
    return actual, sum(k.windows for k, _ in counts.values()), sum(u.windows for _, u in counts.values())


def freeze_v2_configuration(report: dict, source_sha256: str) -> FrozenV2OpenSetConfiguration:
    """Strict report contract and whitelisted output; no IDs/paths/fold thresholds copied."""
    try:
        _finite_tree(report)
        _object(report, V2DiagnosticReport)
        _require(set(report) == {f.name for f in fields(V2DiagnosticReport)}, "incomplete/unsupported V2 report schema")
        for f in fields(V2DiagnosticReport):
            if f.default is not MISSING:
                # JSON pairs become lists; compare canonical structure, not tuple/list identity.
                expected = [list(p) for p in f.default] if f.name in ("identity_model", "score_definitions") else f.default
                _require(report[f.name] == expected, f"unexpected {f.name}")
                if type(f.default) is bool:
                    _require(report[f.name] is f.default, f"invalid {f.name} flag")
        _require(type(report["schema_version"]) is int and type(report["original_feature_count"]) is int
                 and type(report["identity_model_fit_count"]) is int, "invalid report version/dimensions")
        model = dict(report["identity_model"])
        _require(model["probability"] is False and type(model["selected_features"]) is int
                 and type(model["C"]) in (int, float) and type(model["gamma"]) in (int, float), "invalid fixed model settings")
        _require(report["best_v2_development_candidate"] == V2_FROZEN_CANDIDATE, "winner must be svm_top1_global__window_7__normal")
        ids = report["known_identity_ids"]
        _require(isinstance(ids, list) and len(ids) >= 2 and ids == sorted(set(ids)), "invalid known identity metadata")
        for person in ids:
            validate_identifier(person)
        rotations = _integer(report["unknown_development_identity_count"], minimum=2)
        _integer(report["known_source_samples"])
        _integer(report["unknown_source_samples"])
        baseline = _object(report["v1_baseline"], V1Reference)
        for f in fields(V1Reference):
            if not f.name.endswith("sha256"):
                _rate(baseline[f.name])
        provenance = V2ReportProvenance(source_sha256, baseline["report_sha256"], baseline["identity_split_sha256"])
        _require(isinstance(report["folds"], list) and len(report["folds"]) == 3, "missing cross-session fold policy")
        folds = tuple(StudyFold(f["name"], tuple(f["fit_sessions"]), f["calibration_session"]) for f in report["folds"])
        closed = _closed(report, ids)
        configs = configurations()
        _require(isinstance(report["candidates"], list) and len(report["candidates"]) == len(configs), "incomplete strategy results")
        selected = None
        for row, cfg in zip(report["candidates"], configs):
            metrics, known_windows, unknown_windows = _candidate(row, cfg, ids, folds, rotations, report)
            if cfg.identifier == V2_FROZEN_CANDIDATE:
                selected = V2DevelopmentEvidence(closed, row["mean_roc_auc"], row["mean_eer"], metrics,
                                                 report["known_source_samples"], report["unknown_source_samples"], len(ids),
                                                 rotations, known_windows, unknown_windows)
        ranked = [r["identifier"] for r in sorted(report["candidates"], key=rank_key)]
        _require(report["ranking"] == ranked and ranked[0] == V2_FROZEN_CANDIDATE, "declared ranking/winner disagrees with evidence")
        _distributions(report, folds)
        return FrozenV2OpenSetConfiguration(provenance, selected, folds)
    except ConfigurationFreezeError:
        raise
    except Exception:
        raise ConfigurationFreezeError("V2 diagnostic report validation failed: malformed or inconsistent aggregate metadata") from None
