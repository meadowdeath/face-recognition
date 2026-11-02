"""Nested V3 execution. Outer scoring follows selection and threshold freezing."""

from statistics import fmean, pstdev
import re

import numpy as np

from face_recognition.application.create_v3_cv_protocol import metadata_sha256
from face_recognition.application.rejection_metrics import calibrate_threshold
from face_recognition.application.rejection_score_analysis import score_separation
from face_recognition.application.v3_novelty_calibration import (
    CrossFittedWindowPool, binary_metrics, candidate_ranking, evaluate_pair, moments,
    summarize_candidates, transport_diagnostics,
)
from face_recognition.application.v3_novelty_data import ScoredPopulation
from face_recognition.domain.models.classification import ClassificationMetrics
from face_recognition.domain.models.rejection import RejectionMetrics
from face_recognition.domain.models.v3_novelty_evaluation import (
    CANDIDATES, InnerNoveltyResult, OuterCalibration, OuterNoveltyResult, V3NoveltyConfiguration, V3NoveltyReport,
)


class V3NoveltyEvaluator:
    def __init__(self, runner, progress=lambda message: None):
        self.runner, self.progress = runner, progress

    def evaluate(self, data, protocol_file_sha256):
        data.validate()
        if not isinstance(protocol_file_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", protocol_file_sha256):
            raise ValueError("Invalid protocol-file provenance digest")
        protocol, configuration = data.protocol, V3NoveltyConfiguration()
        config_sha = metadata_sha256(configuration)
        identities = len(protocol.specification.known_ids)
        outer_results = []
        for outer in protocol.outer_folds:
            self.progress(f"{outer.name}: inner fitting/calibration/validation")
            cache, inner_results = {}, []
            pools = {method: (CrossFittedWindowPool(), CrossFittedWindowPool()) for method in CANDIDATES}
            for split in outer.inner_splits:
                # Full source-content fingerprint + authoritative role catalog,
                # outer fold, exact fit sessions and fixed complete configuration.
                key = (protocol.protocol_sha256, data.payload_sha256, outer.name, split.known_fit_sessions, config_sha)
                if key not in cache:
                    self.progress(f"{outer.name}: fit KNOWN development population {len(cache) + 1}/3")
                    fit = data.known(split.known_fit_sessions)
                    model = self.runner.fit(fit, identities)
                    known = data.known(tuple(s for s in outer.known_development_sessions if s not in split.known_fit_sessions))
                    unknown = data.unknown(outer.unknown_development_ids)
                    cache[key] = (ScoredPopulation(known, self.runner.infer(model, known)),
                                  ScoredPopulation(unknown, self.runner.infer(model, unknown)))
                    del fit, known, unknown, model
                known_scores, unknown_scores = cache[key]
                for method in CANDIDATES:
                    kc = known_scores.batch(method, sessions=(split.known_calibration_session,))
                    uc = unknown_scores.batch(method, identities=split.unknown_calibration_ids)
                    threshold = calibrate_threshold(kc.scores, kc.correct, uc.scores, allow_negative=True).threshold
                    pools[method][0].add(key, kc)
                    pools[method][1].add(key, uc)
                    kv = known_scores.batch(method, sessions=(split.known_validation_session,))
                    uv = unknown_scores.batch(method, identities=split.unknown_validation_ids)
                    metrics, diagnostics = evaluate_pair(kv, uv, threshold)
                    inner_results.append(InnerNoveltyResult(split.name, method, threshold, len(kc.scores), len(uc.scores),
                                                           kv.counts, uv.counts, metrics, diagnostics))
            summaries = summarize_candidates(inner_results)
            ranking = candidate_ranking(summaries)
            selected = ranking[0]
            for summary in summaries:
                self.progress(f"{outer.name}: {summary.method} inner validation BA={summary.mean_metrics.binary_balanced_accuracy:.6f}")
            cross_known, known_pool = pools[selected][0].finish(expected_models=2)
            cross_unknown, unknown_pool = pools[selected][1].finish(expected_models=3)
            # Identity correctness is deliberately absent from pooled binary
            # threshold construction; it does not affect the BA selection rule.
            calibration = calibrate_threshold(cross_known, np.ones(len(cross_known), dtype=bool), cross_unknown, allow_negative=True)
            frozen_threshold = calibration.threshold
            outer_calibration = OuterCalibration(frozen_threshold, binary_metrics(calibration.metrics), known_pool, unknown_pool,
                                                  score_separation(cross_known, cross_unknown))
            cache_size = len(cache)
            cache.clear()
            del known_scores, unknown_scores, pools
            self.progress(f"{outer.name}: selected {selected}; outer-development threshold frozen; refit all development KNOWN sessions")
            fit = data.known(outer.known_development_sessions)
            final_model = self.runner.fit(fit, identities, methods=(selected,))
            # Training/resubstitution scores below are diagnostics ONLY. No
            # calibration call follows, and outer test is not accessed here.
            final_known = ScoredPopulation(fit, self.runner.infer(final_model, fit)).batch(selected)
            development_unknown = data.unknown(outer.unknown_development_ids)
            final_unknown = ScoredPopulation(development_unknown, self.runner.infer(final_model, development_unknown)).batch(selected)
            transfer = transport_diagnostics(cross_known, cross_unknown, final_known.scores, final_unknown.scores, frozen_threshold)
            del development_unknown, final_known, final_unknown, cross_known, cross_unknown
            # First and only outer-test inference, after all selection/calibration.
            known_test, unknown_test = data.known((outer.known_test_session,)), data.unknown(outer.unknown_test_ids)
            frames = self.runner.infer(final_model, known_test)
            closed = self.runner.classification_metrics(known_test.y, frames.predicted, identities)
            kw = ScoredPopulation(known_test, frames).batch(selected)
            uw = ScoredPopulation(unknown_test, self.runner.infer(final_model, unknown_test)).batch(selected)
            metrics, diagnostics = evaluate_pair(kw, uw, frozen_threshold)
            outer_results.append(OuterNoveltyResult(outer.name, selected, frozen_threshold, len(fit.y), kw.counts, uw.counts,
                        closed, metrics, diagnostics, tuple(inner_results), summaries, ranking, outer_calibration,
                        transfer, cache_size, len(outer.inner_splits) - cache_size))
            self.progress(f"{outer.name}: outer evaluation complete; transport shift warning={transfer.warning}")
            del final_model, fit, known_test, unknown_test, frames, kw, uw
        mean_closed, std_closed = moments((r.closed_set_known_frames for r in outer_results), ClassificationMetrics)
        mean_metrics, std_metrics = moments((r.operational_metrics for r in outer_results), RejectionMetrics)
        auc = [r.diagnostics.roc_auc for r in outer_results]
        eer = [r.diagnostics.equal_error_point.eer for r in outer_results]
        return V3NoveltyReport(protocol.protocol_sha256, protocol_file_sha256, protocol.catalog_sha256,
                protocol.catalog.processing_manifest_sha256, data.payload_sha256, config_sha, identities, len(data.samples),
                tuple(outer_results), mean_closed, std_closed, mean_metrics, std_metrics, fmean(auc), pstdev(auc), fmean(eer), pstdev(eer),
                tuple((method, sum(r.selected_method == method for r in outer_results)) for method in CANDIDATES),
                summarize_candidates(tuple(r for outer in outer_results for r in outer.inner_results)), self.runner.versions)
