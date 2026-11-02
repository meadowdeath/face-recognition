"""Representation alone varies; reuse sealed same-model folds and all metric rules."""

from dataclasses import replace
from statistics import fmean, pstdev
import re

import numpy as np

from face_recognition.application.create_v3_cv_protocol import metadata_sha256
from face_recognition.application.evaluate_v3_mahalanobis_calibration import _FitOnceFold, cyclic_session_roles
from face_recognition.application.rejection_metrics import calibrate_threshold
from face_recognition.application.rejection_score_analysis import score_separation
from face_recognition.application.v3_mahalanobis_breakdowns import aggregate_breakdowns, fold_breakdowns
from face_recognition.application.v3_novelty_calibration import binary_metrics, evaluate_pair, moments
from face_recognition.application.v3_novelty_data import ScoredPopulation, validate_representation_catalog
from face_recognition.domain.models.classification import ClassificationMetrics
from face_recognition.domain.models.dataset import CONDITIONS
from face_recognition.domain.models.processed_sample import BASELINE_REPRESENTATION, POSE_REPRESENTATION
from face_recognition.domain.models.rejection import RejectionError, RejectionMetrics
from face_recognition.domain.models.v3_mahalanobis_calibration import METHOD, MahalanobisCalibrationFold, V3MahalanobisConfiguration
from face_recognition.domain.models.v3_pose_representation import (
    POSE_DELTA_CONDITIONS, ConditionRepresentationComparison, PoseRepresentationFold, RepresentationResult, V3PoseRepresentationReport,
)


def validate_representation_pair(baseline, candidate):
    baseline.validate()
    candidate.validate()
    if (baseline.representation != BASELINE_REPRESENTATION or candidate.representation != POSE_REPRESENTATION
            or baseline.protocol != candidate.protocol or baseline.samples != candidate.samples
            or not np.array_equal(baseline.y, candidate.y)):
        raise RejectionError("Representation populations/provenance must correspond exactly to the unchanged protocol")


def rank_representations(results):
    if {r.representation for r in results} != {BASELINE_REPRESENTATION, POSE_REPRESENTATION} or len(results) != 2:
        raise RejectionError("Exactly baseline and pose candidate are required")
    def key(row):
        values = (row.mean_metrics.binary_balanced_accuracy, row.mean_metrics.known_rejection_rate,
                  row.mean_metrics.unknown_false_acceptance_rate, row.mean_closed_set.macro_f1)
        if not np.isfinite(values).all():
            raise RejectionError("Representation comparison requires finite aggregate metrics")
        return -values[0], values[1], values[2], -values[3], row.representation != BASELINE_REPRESENTATION
    return tuple(r.representation for r in sorted(results, key=key))


class V3PoseRepresentationEvaluator:
    def __init__(self, runner_factory, progress=lambda message: None):
        self.runner_factory, self.progress = runner_factory, progress

    def evaluate(self, baseline, candidate, protocol_file_sha256, candidate_catalog, normalization_definition):
        validate_representation_pair(baseline, candidate)
        if not isinstance(protocol_file_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", protocol_file_sha256):
            raise RejectionError("Invalid protocol-file fingerprint")
        # Catalog validation is also performed by the strict loader. This check
        # keeps direct application callers from attaching unrelated provenance.
        if candidate_catalog.representation != POSE_REPRESENTATION:
            raise RejectionError("Candidate provenance must identify all_xyz_pose_v2")
        validate_representation_catalog(baseline.protocol, candidate_catalog)
        results = []
        for data, catalog in ((baseline, baseline.protocol.catalog), (candidate, candidate_catalog)):
            runner = self.runner_factory()
            rows, breakdowns = [], []
            for roles in cyclic_session_roles(data.protocol):
                self.progress(f"{data.representation} / {roles.fold_name}: fit two KNOWN sessions; fixed Mahalanobis")
                frozen = _FitOnceFold(runner, data, roles)
                known = data.known((roles.known_calibration_session,))
                unknown = data.unknown(roles.unknown_calibration_ids)
                kc = ScoredPopulation(known, frozen.infer(known, calibration=True)).batch(METHOD)
                uc = ScoredPopulation(unknown, frozen.infer(unknown, calibration=True)).batch(METHOD)
                calibration = calibrate_threshold(kc.scores, kc.correct, uc.scores, allow_negative=True)
                frozen.freeze_threshold(calibration.threshold)
                calibration_diagnostics = score_separation(kc.scores, uc.scores)
                del known, unknown
                self.progress(f"{data.representation} / {roles.fold_name}: threshold frozen; same model scores test")
                known, unknown = data.known((roles.known_test_session,)), data.unknown(roles.unknown_test_ids)
                frames = frozen.infer(known, calibration=False)
                closed = runner.classification_metrics(known.y, frames.predicted, len(data.protocol.specification.known_ids))
                kw = ScoredPopulation(known, frames).batch(METHOD)
                uw = ScoredPopulation(unknown, frozen.infer(unknown, calibration=False)).batch(METHOD)
                metrics, diagnostic = evaluate_pair(kw, uw, frozen.threshold)
                row = MahalanobisCalibrationFold(roles, frozen.fit_frames, frozen.threshold, kc.counts, uc.counts,
                    binary_metrics(calibration.metrics), calibration_diagnostics, kw.counts, uw.counts, closed, metrics, diagnostic,
                    frozen.state_sha256, frozen.complete(), 1, True, True)
                breakdown = fold_breakdowns(roles.fold_name, frozen.threshold, kw, uw, roles.known_test_session, roles.unknown_test_ids)
                rows.append(PoseRepresentationFold(row, breakdown.known_per_condition, breakdown.unknown_per_condition,
                                                   breakdown.known_per_session, breakdown.unknown_per_identity))
                breakdowns.append(breakdown)
                del frozen, known, unknown, frames, kc, uc, kw, uw
            mean_closed, std_closed = moments((r.evaluation.closed_set_known_frames for r in rows), ClassificationMetrics)
            mean_metrics, std_metrics = moments((r.evaluation.operational_metrics for r in rows), RejectionMetrics)
            auc = [r.evaluation.diagnostics.roc_auc for r in rows]
            eer = [r.evaluation.diagnostics.equal_error_point.eer for r in rows]
            configuration = replace(V3MahalanobisConfiguration(), representation=data.representation)
            results.append(RepresentationResult(data.representation, metadata_sha256(catalog), catalog.processing_manifest_sha256,
                data.payload_sha256, configuration, metadata_sha256(configuration), tuple(rows), mean_closed, std_closed,
                mean_metrics, std_metrics, fmean(auc), pstdev(auc), fmean(eer), pstdev(eer), aggregate_breakdowns(tuple(breakdowns)), runner.versions))
        if results[0].versions != results[1].versions:
            raise RejectionError("Software configuration differs between representation evaluations")
        tables = [{r.label: r for r in result.diagnostics.known_condition_severity} for result in results]
        comparison = []
        for condition in CONDITIONS:
            first, second = (table[condition].counts for table in tables)
            if first.frr is None or second.frr is None or first.total_windows != second.total_windows:
                raise RejectionError("Every condition, including pitch_down, requires paired complete windows")
            comparison.append(ConditionRepresentationComparison(condition, first.frr, second.frr, second.frr - first.frr,
                                first.correct_identification_rate, second.correct_identification_rate))
        ranking = rank_representations(results)
        deltas = {r.condition: r.frr_delta_candidate_minus_baseline for r in comparison}
        return V3PoseRepresentationReport(baseline.protocol.protocol_sha256, protocol_file_sha256, baseline.protocol.catalog_sha256,
            normalization_definition, metadata_sha256(normalization_definition), tuple(results), tuple(comparison),
            tuple((condition, deltas[condition]) for condition in POSE_DELTA_CONDITIONS), ranking, ranking[0])
