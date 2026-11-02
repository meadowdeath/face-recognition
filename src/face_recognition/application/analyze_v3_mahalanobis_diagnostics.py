"""Post-hoc replay of immutable evidence. No threshold search or selection."""

from dataclasses import replace
from pathlib import PurePosixPath, PureWindowsPath
import re

from face_recognition.application.create_v3_cv_protocol import metadata_sha256
from face_recognition.application.evaluate_v3_mahalanobis_calibration import cyclic_session_roles
from face_recognition.application.rejection_metrics import evaluate_threshold
from face_recognition.application.rejection_score_analysis import score_distribution
from face_recognition.application.v3_mahalanobis_breakdowns import aggregate_breakdowns, fold_breakdowns, interpret_breakdowns
from face_recognition.application.v3_novelty_data import ScoredPopulation
from face_recognition.domain.models.rejection import RejectionError
from face_recognition.domain.models.v3_mahalanobis_calibration import METHOD, THRESHOLD_TRANSPORT, V3MahalanobisConfiguration
from face_recognition.domain.models.v3_mahalanobis_diagnostics import V3MahalanobisDiagnosticsReport


class V3MahalanobisDiagnosticAnalyzer:
    def __init__(self, runner, progress=lambda message: None):
        self.runner, self.progress = runner, progress

    def analyze(self, data, evidence, provenance):
        data.validate()
        config_sha = metadata_sha256(V3MahalanobisConfiguration())
        if (evidence.schema_version != 1 or evidence.completion_state != "completed"
                or evidence.threshold_transport != THRESHOLD_TRANSPORT or metadata_sha256(evidence.configuration) != config_sha
                or evidence.configuration_sha256 != config_sha or evidence.protocol_sha256 != data.protocol.protocol_sha256
                or evidence.catalog_sha256 != data.protocol.catalog_sha256 or evidence.payload_sha256 != data.payload_sha256
                or evidence.processing_manifest_sha256 != data.protocol.catalog.processing_manifest_sha256
                or evidence.source_samples != len(data.samples) or evidence.identity_count != len(data.protocol.specification.known_ids)
                or evidence.versions != self.runner.versions
                or tuple(row.roles for row in evidence.outer_results) != cyclic_session_roles(data.protocol)):
            raise RejectionError("Diagnostic replay does not match immutable study/data/environment")
        expected = {"protocol_sha256": evidence.protocol_sha256, "protocol_file_sha256": evidence.protocol_file_sha256,
                    "initial_selection_report_sha256": evidence.selection_provenance.report_sha256,
                    "catalog_sha256": evidence.catalog_sha256, "processing_manifest_sha256": evidence.processing_manifest_sha256,
                    "payload_sha256": evidence.payload_sha256, "configuration_sha256": config_sha}
        if (any(getattr(provenance, name) != value for name, value in expected.items())
                or not re.fullmatch(r"[a-f0-9]{64}", provenance.evaluation_report_sha256)):
            raise RejectionError("Diagnostic provenance mismatch")
        for reference in (provenance.protocol_reference, provenance.evaluation_report_reference):
            if not reference or reference in (".", "..") or PurePosixPath(reference).name != reference or PureWindowsPath(reference).name != reference:
                raise RejectionError("Provenance stores technical filenames only, never absolute source paths")
        folds, seen_known, seen_unknown = [], set(), set()
        for saved in evidence.outer_results:
            self.progress(f"{saved.roles.fold_name}: reconstruct original fit state; no calibration/selection")
            if (saved.model_sha256_before_calibration != saved.model_sha256_after_test or saved.fit_calls != 1
                    or not saved.same_model_identity_verified or not saved.no_refit_after_calibration):
                raise RejectionError("Source model was not frozen consistently")
            fit = data.known(saved.roles.known_fit_sessions)
            if len(fit.samples) != saved.fit_known_frames:
                raise RejectionError("Saved fit population count mismatch")
            before = self.runner.fit_count
            model = self.runner.fit(fit, evidence.identity_count, methods=(METHOD,))
            generation, identity = self.runner.fit_count, self.runner.model_identity(model)
            expected_state = saved.model_sha256_before_calibration
            def verify():
                if (generation != before + 1 or self.runner.fit_count != generation or self.runner.model_identity(model) != identity
                        or self.runner.model_state_sha256(model) != expected_state):
                    raise RejectionError("Reconstructed model differs from saved evidence or changed during diagnostics")
            verify()  # Before any outer-test scoring; never substitute a different fit.
            del fit
            known, unknown = data.known((saved.roles.known_test_session,)), data.unknown(saved.roles.unknown_test_ids)
            kframes = self.runner.infer(model, known)
            verify()
            uframes = self.runner.infer(model, unknown)
            verify()
            if any(tuple(name for name, _ in frames.confidence) != (METHOD,) for frames in (kframes, uframes)):
                raise RejectionError("Diagnostic replay is Mahalanobis only")
            kw = ScoredPopulation(known, kframes).batch(METHOD)
            uw = ScoredPopulation(unknown, uframes).batch(METHOD)
            # Descriptive checks, NOT recalibration. No ROC/EER threshold
            # enumeration is performed anywhere in this diagnostic workflow.
            metrics = evaluate_threshold(kw.scores, kw.correct, uw.scores, saved.threshold, allow_negative=True)
            closed = self.runner.classification_metrics(known.y, kframes.predicted, evidence.identity_count)
            if (kw.counts != saved.known_test_counts or uw.counts != saved.unknown_test_counts
                    or metrics != saved.operational_metrics or closed != saved.closed_set_known_frames
                    or score_distribution(kw.scores) != saved.diagnostics.known_distribution
                    or score_distribution(uw.scores) != saved.diagnostics.unknown_distribution):
                raise RejectionError("Replayed outer summaries differ from saved evaluation; refusing new diagnostic evidence")
            if seen_known.intersection(kw.keys) or seen_unknown.intersection(uw.keys):
                raise RejectionError("Physical outer-test window repeated across diagnostic folds")
            seen_known.update(kw.keys); seen_unknown.update(uw.keys)
            folds.append(replace(fold_breakdowns(saved.roles.fold_name, saved.threshold, kw, uw,
                                        saved.roles.known_test_session, saved.roles.unknown_test_ids),
                                 saved_model_state_verified=True, saved_outer_summaries_verified=True))
            verify()
            self.progress(f"{saved.roles.fold_name}: saved state, threshold, counts and summaries verified")
            del model, known, unknown, kframes, uframes, kw, uw
        aggregate = aggregate_breakdowns(tuple(folds))
        return V3MahalanobisDiagnosticsReport(provenance, tuple(folds), aggregate, interpret_breakdowns(aggregate), self.runner.versions)
