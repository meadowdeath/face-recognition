"""Validate aggregate development evidence and freeze choices; no fitting or data loading."""

from dataclasses import fields
from math import isclose, isfinite
from statistics import fmean, pstdev
from typing import Any

from face_recognition.domain.models.open_set_config import (
    DEVELOPMENT_EXPERIMENT, FROZEN_CANDIDATE, ConfigurationFreezeError, DevelopmentReportProvenance,
    FrozenDevelopmentEvidence, FrozenOpenSetConfiguration, OpenSetArchitecture,
)
from face_recognition.domain.models.rejection_score_evaluation import EqualErrorPoint, ScoreAnalysis
from face_recognition.domain.models.temporal_distance_novelty import TEMPORAL_DISTANCE_NEIGHBORS, TemporalDistanceNoveltyReport
from face_recognition.domain.models.temporal_open_set import WINDOW_SIZES, TemporalFoldResult, TemporalResult, WindowCounts
from face_recognition.domain.models.unknown_rejection import RejectionMetrics, ThresholdCalibration


def _require(condition: bool, category: str) -> None:
    if not condition:
        raise ConfigurationFreezeError(f"Development report validation failed: {category}")


def _object(value: Any, model: type) -> dict:
    _require(isinstance(value, dict) and {f.name for f in fields(model)} <= value.keys(), f"incomplete {model.__name__}")
    return value


def _finite_tree(value: Any) -> None:
    if isinstance(value, dict):
        _require(not set(value) & {"landmarks", "features", "feature_vectors", "support_vectors_", "reference_matrix",
                                   "nearest_indices", "nearest_labels", "per_sample_scores"}, "unexpected sample/vector payload")
        for child in value.values():
            _finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            _finite_tree(child)
    elif type(value) in (float, int):
        _require(isfinite(value), "non-finite numeric metadata")


def _rate(value: Any) -> float:
    _require(type(value) in (int, float) and isfinite(value) and 0 <= value <= 1, "invalid evidence rate")
    return float(value)


def _close(a: float, b: float) -> bool:
    return isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def _metrics(value: Any, *, relationships: bool = True) -> RejectionMetrics:
    value = _object(value, RejectionMetrics)
    result = RejectionMetrics(**{f.name: _rate(value[f.name]) for f in fields(RejectionMetrics)})
    if relationships:
        _require(_close(result.known_acceptance_rate + result.known_rejection_rate, 1)
                 and _close(result.unknown_rejection_rate + result.unknown_false_acceptance_rate, 1)
                 and _close(result.known_correct_identification_rate + result.known_misidentification_rate, result.known_acceptance_rate)
                 and _close(result.binary_balanced_accuracy, (result.known_acceptance_rate + result.unknown_rejection_rate) / 2),
                 "inconsistent rate definitions")
    return result


def _counts(value: Any, size: int) -> WindowCounts:
    value = _object(value, WindowCounts)
    _require(all(type(value[f.name]) is int and value[f.name] >= 0 for f in fields(WindowCounts)), "invalid window counts")
    result = WindowCounts(**{f.name: value[f.name] for f in fields(WindowCounts)})
    _require(result.windows > 0 and result.used_samples == result.windows * size
             and result.used_samples + result.discarded_samples == result.source_samples, "inconsistent window counts")
    return result


def _fold(value: Any, size: int) -> tuple[float, float, RejectionMetrics, WindowCounts, WindowCounts]:
    fold = _object(value, TemporalFoldResult)
    known, unknown = _counts(fold["known_counts"], size), _counts(fold["unknown_counts"], size)
    _require(type(fold["fit_samples"]) is int and fold["fit_samples"] >= 5, "invalid fit sample count")
    _require(isinstance(fold["fold_name"], str) and bool(fold["fold_name"]), "missing fold name")
    for key in ("fit_session_ids", "validation_session_ids"):
        _require(isinstance(fold[key], list) and len(fold[key]) == 1 and isinstance(fold[key][0], str), "incomplete session protocol")
    _require(set(fold["fit_session_ids"]).isdisjoint(fold["validation_session_ids"]), "overlapping fit/validation sessions")
    _require(isinstance(fold["conditions"], list) and bool(fold["conditions"]), "missing condition diagnostics")
    analysis = _object(fold["analysis"], ScoreAnalysis)
    roc_auc = _rate(analysis["roc_auc"])
    point = _object(analysis["equal_error_point"], EqualErrorPoint)
    eer = _rate(point["eer"])
    _require(_close(eer, (_rate(point["far"]) + _rate(point["frr"])) / 2), "inconsistent EER definition")
    _require(type(point["threshold"]) in (int, float), "missing EER threshold")
    calibration = _object(analysis["calibration"], ThresholdCalibration)
    _require(type(calibration["threshold"]) in (int, float) and type(calibration["thresholds_evaluated"]) is int
             and calibration["thresholds_evaluated"] > 0, "incomplete threshold calibration")
    metrics = _metrics(calibration["metrics"])
    _require(_close(metrics.open_set_exact_accuracy,
                    (metrics.known_correct_identification_rate * known.windows + metrics.unknown_rejection_rate * unknown.windows)
                    / (known.windows + unknown.windows)), "inconsistent open-set accuracy")
    _require(isinstance(analysis["far_operating_points"], list) and len(analysis["far_operating_points"]) == 3,
             "missing FAR diagnostics")
    _metrics(fold["at_development_threshold"])
    return roc_auc, eer, metrics, known, unknown


def freeze_development_configuration(report: dict, source_sha256: str) -> FrozenOpenSetConfiguration:
    """Whitelist only aggregate evidence; never copy source IDs, paths or fold thresholds.

    Historical performance numbers are deliberately absent: evidence comes
    from the validated report and is checked against that report's fold means.
    """
    try:
        _finite_tree(report)
        report = _object(report, TemporalDistanceNoveltyReport)
        architecture = OpenSetArchitecture()
        _require(type(report["schema_version"]) is int and report["schema_version"] == 1
                 and report["experiment"] == DEVELOPMENT_EXPERIMENT and report["completion_state"] == "completed",
                 "unsupported or incomplete source report")
        _require(report["source_representation"] == architecture.source_representation
                 and type(report["original_feature_count"]) is int and report["original_feature_count"] == architecture.original_feature_count,
                 "unexpected representation or original feature count")
        _require(report["model"] == {"k": 448, "C": 4.0, "gamma": .001, "kernel": "rbf", "probability": False},
                 "expected ANOVA-448 and fixed RBF SVM C=4 gamma=0.001")
        _require(report["configuration"] == {"name": "SVM", "classifier_parameters": dict(architecture.svm_parameters),
                                              "scaler_parameters": dict(architecture.scaler_parameters)}, "inconsistent classifier/scaler configuration")
        _require(report["model"]["probability"] is False and report["configuration"]["classifier_parameters"]["probability"] is False
                 and all(report["configuration"]["scaler_parameters"][name] is value for name, value in architecture.scaler_parameters),
                 "invalid classifier/scaler boolean settings")
        _require(report["known_test_payloads_accessed"] is False and report["unknown_holdout_payloads_accessed"] is False,
                 "source report does not protect final holdouts")
        _require(report["window_sizes"] == list(WINDOW_SIZES) and report["neighbor_counts"] == dict(TEMPORAL_DISTANCE_NEIGHBORS),
                 "incomplete temporal experiment")
        # Version 1 report prose is part of its contract. Do not attest a median,
        # voting, distance, or threshold policy when its source says otherwise.
        for name, definition in TemporalDistanceNoveltyReport.__dataclass_fields__.items():
            if name.endswith("policy") or name in ("distance_space", "median_rule", "scope", "score_definitions"):
                expected = dict(definition.default) if name == "score_definitions" else definition.default
                _require(report[name] == expected, f"unsupported {name}")
        _require(report["best_temporal_distance_novelty_development_candidate"] == FROZEN_CANDIDATE,
                 "selected candidate must be mean_5nn with window 7")
        expected_rows = [(family, size) for family, _ in TEMPORAL_DISTANCE_NEIGHBORS for size in WINDOW_SIZES]
        _require(isinstance(report["results"], list) and len(report["results"]) == len(expected_rows), "incomplete configuration results")
        summaries, selected = [], None
        for value, (family, size) in zip(report["results"], expected_rows):
            row = _object(value, TemporalResult)
            identifier = f"{family}__window_{size}"
            _require(row["score_family"] == family and type(row["window_size"]) is int and row["window_size"] == size
                     and row.get("configuration_id") == identifier, "unexpected method/window configuration")
            _require(isinstance(row["folds"], list) and len(row["folds"]) == 2 and row["vs_window1"] is not None,
                     "incomplete fold/reference results")
            fold_data = [_fold(fold, size) for fold in row["folds"]]
            a, b = row["folds"]
            _require(a["fold_name"] != b["fold_name"] and a["fit_session_ids"] == b["validation_session_ids"]
                     and b["fit_session_ids"] == a["validation_session_ids"], "expected opposite-session development folds")
            mean_metrics, std_metrics = _metrics(row["mean_metrics"]), _metrics(row["std_metrics"], relationships=False)
            for model, statistic in ((mean_metrics, fmean), (std_metrics, pstdev)):
                _require(all(_close(getattr(model, f.name), statistic(getattr(fold[2], f.name) for fold in fold_data))
                             for f in fields(RejectionMetrics)), "aggregate metrics disagree with fold evidence")
            for name, column, statistic in (("mean_roc_auc", 0, fmean), ("std_roc_auc", 0, pstdev),
                                             ("mean_eer", 1, fmean), ("std_eer", 1, pstdev)):
                _require(_close(_rate(row[name]), statistic(fold[column] for fold in fold_data)), "aggregate AUC/EER disagrees with folds")
            known, unknown = _counts(row["total_known_counts"], size), _counts(row["total_unknown_counts"], size)
            for total, column in ((known, 3), (unknown, 4)):
                _require(all(getattr(total, f.name) == sum(getattr(fold[column], f.name) for fold in fold_data)
                             for f in fields(WindowCounts)), "aggregate window counts disagree with folds")
            summaries.append((identifier, family, size, row["mean_roc_auc"], row["mean_eer"], mean_metrics))
            if identifier == FROZEN_CANDIDATE:
                selected = FrozenDevelopmentEvidence(float(row["mean_roc_auc"]), float(row["mean_eer"]), mean_metrics,
                                                       known.windows, unknown.windows)
        counts = dict(TEMPORAL_DISTANCE_NEIGHBORS)
        ranked = sorted(summaries, key=lambda r: (-r[3], r[4], r[5].unknown_false_acceptance_rate,
                                                 -r[5].known_correct_identification_rate, -r[5].open_set_exact_accuracy,
                                                 r[5].known_rejection_rate, r[2], counts[r[1]]))
        _require(report["ranking"] == [row[0] for row in ranked] and report["ranking"][0] == FROZEN_CANDIDATE,
                 "selected winner/ranking disagrees with evidence")
        return FrozenOpenSetConfiguration(DevelopmentReportProvenance(source_sha256), selected)
    except ConfigurationFreezeError:
        raise
    except Exception:
        raise ConfigurationFreezeError("Development report validation failed: malformed or inconsistent aggregate metadata") from None
