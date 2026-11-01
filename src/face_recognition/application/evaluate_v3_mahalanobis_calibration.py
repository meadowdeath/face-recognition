"""Fixed-method calibration refinement: fit once, calibrate, test unchanged."""

from statistics import fmean, pstdev
from math import isfinite
import re

from face_recognition.application.create_v3_cv_protocol import metadata_sha256, validate_v3_cv_protocol
from face_recognition.application.rejection_metrics import calibrate_threshold
from face_recognition.application.rejection_score_analysis import score_separation
from face_recognition.application.v3_novelty_calibration import binary_metrics, evaluate_pair, moments
from face_recognition.application.v3_novelty_data import ScoredPopulation
from face_recognition.domain.models.classification import ClassificationMetrics
from face_recognition.domain.models.rejection import RejectionError, RejectionMetrics
from face_recognition.domain.models.v3_mahalanobis_calibration import (
    METHOD, CyclicFoldRoles, MahalanobisCalibrationFold, V3MahalanobisCalibrationReport, V3MahalanobisConfiguration,
)
from face_recognition.domain.models.v3_novelty_evaluation import V3NoveltyConfiguration


def cyclic_session_roles(protocol):
    """Derive roles from the authoritative ordered specification, never ID literals."""
    validate_v3_cv_protocol(protocol)
    sessions = protocol.specification.known_sessions
    roles = []
    for outer in protocol.outer_folds:
        calibration = sessions[(sessions.index(outer.known_test_session) + 1) % len(sessions)]
        fit = tuple(s for s in sessions if s not in (calibration, outer.known_test_session))
        if len(fit) != 2 or set(fit) | {calibration} != set(outer.known_development_sessions):
            raise RejectionError("Invalid cyclic V3 session partition")
        roles.append(CyclicFoldRoles(outer.name, fit, calibration, outer.known_test_session,
                                     outer.unknown_development_ids, outer.unknown_test_ids))
    return tuple(roles)


def validate_selection_provenance(selection, protocol, payload_sha256=None):
    expected = tuple((outer.name, METHOD) for outer in protocol.outer_folds)
    if (selection.protocol_sha256 != protocol.protocol_sha256 or selection.catalog_sha256 != protocol.catalog_sha256
            or selection.configuration_sha256 != metadata_sha256(V3NoveltyConfiguration())
            or selection.selected_methods != expected or selection.identity_count != len(protocol.specification.known_ids)
            or selection.source_samples != protocol.catalog.declared_source_samples
            or selection.experiment != "v3_nested_novelty_comparison_v1" or type(selection.schema_version) is not int
            or selection.schema_version != 1 or not isinstance(selection.report_sha256, str)
            or not re.fullmatch(r"[a-f0-9]{64}", selection.report_sha256)
            or not isinstance(selection.payload_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", selection.payload_sha256)
            or (payload_sha256 is not None and selection.payload_sha256 != payload_sha256)):
        raise RejectionError("Selection report does not authorize this fixed-method study/data")


class _FitOnceFold:
    """Sealed scoring session: no fit API after construction, threshold set once.

    Fit-generation, component identities and all learned-state bytes are checked
    before/after inference. This rejects refits through the runner even if a
    repeated fit reproduces identical parameters, and rejects state mutation or
    replacement even when no runner fit call was recorded.
    """
    def __init__(self, runner, data, roles):
        self.runner, self.roles = runner, roles
        self.known_ids = data.protocol.specification.known_ids
        self.unknown_sessions = data.protocol.specification.unknown_sessions
        fit = data.known(roles.known_fit_sessions)
        self._authorize(fit, roles.known_fit_sessions, ())
        before = runner.fit_count
        self._model = runner.fit(fit, len(self.known_ids), methods=(METHOD,))
        if runner.fit_count != before + 1:
            raise RejectionError("Exactly one fitting call per calibration fold is required")
        self._generation = runner.fit_count
        self._identity = runner.model_identity(self._model)
        self.state_sha256 = runner.model_state_sha256(self._model)
        if not isinstance(self.state_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", self.state_sha256):
            raise RejectionError("Invalid frozen-model integrity digest")
        self.fit_frames = len(fit.samples)
        self._threshold = None
        self._calibration_roles, self._test_roles = set(), set()

    def _authorize(self, population, known_sessions, unknown_ids):
        people = {s.person_id for s in population.samples}
        sessions = {s.session_id for s in population.samples}
        if people == set(self.known_ids):
            valid = sessions == set(known_sessions) and bool(known_sessions) and (population.y >= 0).all()
            role = "known"
        else:
            valid = people == set(unknown_ids) and bool(unknown_ids) and sessions == set(self.unknown_sessions) and (population.y == -1).all()
            role = "unknown"
        if not valid:
            raise RejectionError("Population is not authorized for this same-model scoring phase")
        return role

    def verify(self):
        if (self.runner.fit_count != self._generation or self.runner.model_identity(self._model) != self._identity
                or self.runner.model_state_sha256(self._model) != self.state_sha256):
            raise RejectionError("Frozen fold was refitted, replaced or mutated during calibration/test")

    def infer(self, population, *, calibration):
        if calibration:
            if self._threshold is not None:
                raise RejectionError("Calibration scoring is already complete")
            role = self._authorize(population, (self.roles.known_calibration_session,), self.roles.unknown_calibration_ids)
            seen = self._calibration_roles
        else:
            if self._threshold is None:
                raise RejectionError("Outer scoring requires a frozen threshold and runs once per role")
            role = self._authorize(population, (self.roles.known_test_session,), self.roles.unknown_test_ids)
            seen = self._test_roles
        if role in seen:
            raise RejectionError("This population role was already scored in this fold phase")
        seen.add(role)
        self.verify()
        result = self.runner.infer(self._model, population)
        self.verify()
        if tuple(name for name, _ in result.confidence) != (METHOD,):
            raise RejectionError("Only Mahalanobis confidence is allowed in this study")
        return result

    def freeze_threshold(self, threshold):
        if self._threshold is not None or self._calibration_roles != {"known", "unknown"} or not isfinite(threshold):
            raise RejectionError("Require complete calibration and one finite threshold")
        self.verify()
        self._threshold = threshold

    @property
    def threshold(self):
        if self._threshold is None:
            raise RejectionError("Threshold has not been calibrated")
        return self._threshold

    def complete(self):
        self.verify()
        if self._calibration_roles != {"known", "unknown"} or self._test_roles != {"known", "unknown"}:
            raise RejectionError("Incomplete same-model calibration/test execution")
        return self.runner.model_state_sha256(self._model)


class V3MahalanobisCalibrationEvaluator:
    def __init__(self, runner, progress=lambda message: None):
        self.runner, self.progress = runner, progress

    def evaluate(self, data, protocol_file_sha256, selection):
        data.validate()
        if not isinstance(protocol_file_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", protocol_file_sha256):
            raise RejectionError("Invalid protocol-file provenance")
        validate_selection_provenance(selection, data.protocol, data.payload_sha256)
        results = []
        for roles in cyclic_session_roles(data.protocol):
            self.progress(f"{roles.fold_name}: fit two KNOWN sessions; fixed Mahalanobis only")
            frozen = _FitOnceFold(self.runner, data, roles)
            known = data.known((roles.known_calibration_session,))
            unknown = data.unknown(roles.unknown_calibration_ids)
            kc = ScoredPopulation(known, frozen.infer(known, calibration=True)).batch(METHOD)
            uc = ScoredPopulation(unknown, frozen.infer(unknown, calibration=True)).batch(METHOD)
            calibration = calibrate_threshold(kc.scores, kc.correct, uc.scores, allow_negative=True)
            frozen.freeze_threshold(calibration.threshold)
            calibration_diagnostics = score_separation(kc.scores, uc.scores)
            del known, unknown
            self.progress(f"{roles.fold_name}: window-level threshold frozen; score outer test with the same model, no refit")
            known = data.known((roles.known_test_session,))
            unknown = data.unknown(roles.unknown_test_ids)
            frames = frozen.infer(known, calibration=False)
            closed = self.runner.classification_metrics(known.y, frames.predicted, len(data.protocol.specification.known_ids))
            kw = ScoredPopulation(known, frames).batch(METHOD)
            uw = ScoredPopulation(unknown, frozen.infer(unknown, calibration=False)).batch(METHOD)
            metrics, diagnostics = evaluate_pair(kw, uw, frozen.threshold)
            after_test = frozen.complete()
            results.append(MahalanobisCalibrationFold(roles, frozen.fit_frames, frozen.threshold, kc.counts, uc.counts,
                binary_metrics(calibration.metrics), calibration_diagnostics, kw.counts, uw.counts, closed, metrics, diagnostics,
                frozen.state_sha256, after_test, 1, True, True))
            self.progress(f"{roles.fold_name}: complete; threshold_transport=none_same_model_calibration; fit calls=1")
            del frozen, known, unknown, frames, kc, uc, kw, uw
        mean_closed, std_closed = moments((r.closed_set_known_frames for r in results), ClassificationMetrics)
        mean_metrics, std_metrics = moments((r.operational_metrics for r in results), RejectionMetrics)
        auc = [r.diagnostics.roc_auc for r in results]
        eer = [r.diagnostics.equal_error_point.eer for r in results]
        far = [r.operational_metrics.unknown_false_acceptance_rate for r in results]
        frr = [r.operational_metrics.known_rejection_rate for r in results]
        config = V3MahalanobisConfiguration()
        return V3MahalanobisCalibrationReport(data.protocol.protocol_sha256, protocol_file_sha256, data.protocol.catalog_sha256,
            data.protocol.catalog.processing_manifest_sha256, data.payload_sha256, metadata_sha256(config), selection,
            len(data.protocol.specification.known_ids), len(data.samples), tuple(results), mean_closed, std_closed, mean_metrics,
            std_metrics, fmean(auc), pstdev(auc), fmean(eer), pstdev(eer), (min(far), max(far)), (min(frr), max(frr)), self.runner.versions)
