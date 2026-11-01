"""Deterministic aggregate-only V3 JSON, atomic external no-overwrite publication."""

from dataclasses import asdict
import json

from face_recognition.application.create_v3_cv_protocol import metadata_sha256
from face_recognition.domain.models.v3_novelty_evaluation import CANDIDATES, V3NoveltyConfiguration
from face_recognition.infrastructure.persistence.aggregate_report import write_aggregate_report_json


def v3_novelty_report_to_json(report):
    if (report.completion_state != "completed" or report.configuration != V3NoveltyConfiguration()
            or report.configuration_sha256 != metadata_sha256(report.configuration)
            or len(report.outer_results) != 4
            or any(len(r.inner_results) != 72 or r.selected_method not in CANDIDATES
                   or r.selected_method != r.inner_ranking[0] or r.threshold != r.calibration.threshold for r in report.outer_results)):
        raise ValueError("Incomplete/inconsistent V3 novelty report")
    return json.dumps(asdict(report), sort_keys=True, allow_nan=False, indent=2) + "\n"


def write_v3_novelty_report(report, output, inputs):
    write_aggregate_report_json(v3_novelty_report_to_json(report), output, inputs)
