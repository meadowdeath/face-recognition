"""One chronological fit and same-state temporal calibration, never evaluation."""

from collections import Counter
from math import isfinite

import numpy as np

from face_recognition.application.freeze_v3_open_set_config import validate_frozen_v3_configuration
from face_recognition.application.rejection_metrics import calibrate_threshold
from face_recognition.application.rejection_score_analysis import score_separation
from face_recognition.application.v3_novelty_calibration import binary_metrics
from face_recognition.application.v3_novelty_data import ScoredPopulation
from face_recognition.domain.models.dataset import CONDITIONS
from face_recognition.domain.models.rejection import RejectionError
from face_recognition.domain.models.v3_deployable_model import (
    DeployableV3Model, V3CalibrationSummary, V3DeployableAllocation, V3TrainingResult,
)


def deployment_allocation(data, config):
    """Use fingerprinted membership, first three chronological sessions and latest fourth.

    Uniform per-condition counts come from the frozen protocol's declared source
    count, then every actual group is checked. For the real catalog this is 20;
    no runtime CLI overrides or performance-based population choices exist.
    """
    validate_frozen_v3_configuration(config)
    data.validate()
    protocol = data.protocol
    if (data.representation != config.pipeline.representation
            or protocol.protocol_sha256 != config.protocol_sha256 or protocol.catalog_sha256 != config.protocol_catalog_sha256):
        raise RejectionError("Deployable data do not match frozen pose/protocol provenance")
    spec = protocol.specification
    group_count = len(protocol.catalog.condition_files)
    n, remainder = divmod(protocol.catalog.declared_source_samples, group_count)
    if remainder or n < config.temporal.window_size:
        raise RejectionError("Uniform complete condition captures are required")
    groups = Counter((s.person_id, s.session_id, s.condition) for s in data.samples)
    if len(groups) != group_count or any(count != n for count in groups.values()):
        raise RejectionError("Actual per-person/session/condition counts differ from the frozen catalog")
    return V3DeployableAllocation(spec.known_ids, spec.known_sessions[:3], spec.known_sessions[3],
        spec.unknown_ids, spec.unknown_sessions, n)


class TrainV3DeployableModel:
    def __init__(self, runner, progress=lambda message: None):
        self.runner, self.progress = runner, progress

    def train(self, data, config, provenance):
        allocation = deployment_allocation(data, config)
        refs = {r.kind: r for r in config.evidence}
        if (provenance.payload_sha256 != data.payload_sha256
                or provenance.protocol_file_sha256 != refs["protocol"].sha256
                or provenance.pose_report_file_sha256 != refs["pose_report"].sha256):
            raise RejectionError("Payload fingerprint mismatch")
        known_fit = data.known(allocation.known_fit_sessions)
        known_cal = data.known((allocation.known_calibration_session,))
        unknown_cal = data.unknown(allocation.unknown_calibration_identities)
        per_session = len(CONDITIONS) * allocation.samples_per_condition
        expected_fit = len(allocation.known_identities) * 3 * per_session
        if (len(known_fit.y) != expected_fit or len(known_cal.y) != len(allocation.known_identities) * per_session
                or len(unknown_cal.y) != len(allocation.unknown_calibration_identities) * 2 * per_session
                or np.any(known_fit.y < 0) or set(s.session_id for s in known_fit.samples) != set(allocation.known_fit_sessions)
                or set(s.session_id for s in known_cal.samples) != {allocation.known_calibration_session}
                or set(s.person_id for s in unknown_cal.samples) != set(allocation.unknown_calibration_identities)
                or set(s.session_id for s in unknown_cal.samples) != set(allocation.unknown_calibration_sessions)
                or self.runner.fit_count != 0):
            raise RejectionError("Invalid fit/calibration role isolation or counts")
        self.progress("Fit once: first three KNOWN sessions only; UNKNOWN and latest KNOWN session excluded")
        fitted = self.runner.fit(known_fit, len(allocation.known_identities), methods=(config.pipeline.novelty_method,))
        generation = self.runner.fit_count
        identity = self.runner.model_identity(fitted)
        before = self.runner.fingerprints(fitted)
        def verify():
            if (generation != 1 or self.runner.fit_count != generation or self.runner.model_identity(fitted) != identity
                    or self.runner.fingerprints(fitted) != before):
                raise RejectionError("Deployable state refitted, replaced or changed after fitting")
        verify()
        class_counts = tuple(int((known_fit.y == i).sum()) for i in range(len(allocation.known_identities)))
        del known_fit
        self.progress("Fitted state sealed; score KNOWN/UNKNOWN calibration with this exact state")
        kframes = self.runner.infer(fitted, known_cal)
        verify()
        uframes = self.runner.infer(fitted, unknown_cal)
        verify()
        if any(tuple(name for name, _ in f.confidence) != (config.pipeline.novelty_method,) for f in (kframes, uframes)):
            raise RejectionError("Only frozen Mahalanobis scores may calibrate deployment")
        kw = ScoredPopulation(known_cal, kframes).batch(config.pipeline.novelty_method)
        uw = ScoredPopulation(unknown_cal, uframes).batch(config.pipeline.novelty_method)
        self.progress("Calibrate one threshold from window-7 median scores; no refit follows")
        calibration = calibrate_threshold(kw.scores, kw.correct, uw.scores, allow_negative=True)
        if type(calibration.threshold) not in (int, float) or not isfinite(calibration.threshold):
            raise RejectionError("Operational threshold must be finite")
        diagnostics = score_separation(kw.scores, uw.scores)
        verify()
        model = DeployableV3Model(fitted, config, allocation.known_identities, float(calibration.threshold))
        summary = V3CalibrationSummary(allocation, provenance, expected_fit, class_counts, kw.counts, uw.counts,
            model.threshold, binary_metrics(calibration.metrics), diagnostics, before, self.runner.fingerprints(fitted),
            generation, self.runner.fit_count, self.runner.support_vector_count(fitted), self.runner.versions)
        verify()
        return V3TrainingResult(model, summary)
