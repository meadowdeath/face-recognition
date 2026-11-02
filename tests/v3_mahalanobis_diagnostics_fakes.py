"""Synthetic historical evidence and deterministic diagnostic replay observations."""

import numpy as np

from face_recognition.application.evaluate_v3_mahalanobis_calibration import V3MahalanobisCalibrationEvaluator
from face_recognition.application.v3_novelty_data import FrameScores
from face_recognition.domain.models.v3_mahalanobis_diagnostics import MahalanobisDiagnosticProvenance
from face_recognition.infrastructure.persistence.artifact_integrity import file_sha256
from face_recognition.infrastructure.persistence.v3_cv_protocol_repository import write_v3_cv_protocol
from face_recognition.infrastructure.persistence.v3_mahalanobis_calibration_report import (
    read_mahalanobis_selection_report, write_mahalanobis_calibration_report,
)
from tests.v3_mahalanobis_fakes import RecordingSameModelRunner, write_selection_report
from tests.v3_novelty_fakes import novelty_data


class DiagnosticReplayRunner(RecordingSameModelRunner):
    def infer(self, model, population):
        frames = super().infer(model, population)
        known = population.y >= 0
        scores = frames.confidence[0][1].copy()
        predicted = frames.predicted.copy()
        for i, sample in enumerate(population.samples):
            if known[i] and (sample.condition == "pitch_up" or (sample.session_id == "capture_d" and sample.condition == "pitch_down")):
                scores[i] = -10. + model.offset
            if not known[i] and sample.person_id.endswith("00"):
                scores[i] = -1. + model.offset
            if known[i] and sample.condition == "yaw_left" and population.y[i] == 1:
                predicted[i] = 0
        return FrameScores(predicted, np.eye(2)[predicted] * 3, (("mahalanobis_distance", scores),))


def diagnostic_fixture(folder, runner=None):
    data, processed, spec_path = novelty_data(folder)
    protocol_path = folder / "protocol.json"
    write_v3_cv_protocol(data.protocol, protocol_path, (processed, spec_path))
    selection_path = write_selection_report(data, folder / "selection.json")
    selection = read_mahalanobis_selection_report(selection_path, data.protocol)
    # Historical synthetic study generation is outside diagnostic execution.
    source = V3MahalanobisCalibrationEvaluator(runner or DiagnosticReplayRunner()).evaluate(data, file_sha256(protocol_path), selection)
    source_path = folder / "evaluation.json"
    write_mahalanobis_calibration_report(source, source_path, (processed, protocol_path, selection_path))
    provenance = MahalanobisDiagnosticProvenance(protocol_path.name, data.protocol.protocol_sha256, file_sha256(protocol_path),
        source_path.name, file_sha256(source_path), selection.report_sha256, data.protocol.catalog_sha256,
        data.protocol.catalog.processing_manifest_sha256, data.payload_sha256, source.configuration_sha256)
    return data, processed, protocol_path, source_path, source, provenance
