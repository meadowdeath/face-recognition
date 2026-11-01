"""External aggregate-only V2 publication, with no mutation/overwrite option."""

from dataclasses import asdict
import json

from face_recognition.application.v2_rejection import configurations
from face_recognition.infrastructure.persistence.baseline_evaluation_report import BaselineReportError, write_aggregate_report_json


def v2_report_to_json(report):
    expected = {c.identifier for c in configurations()}
    if (report.schema_version != 1 or report.status != "development_diagnostics_completed"
            or report.v2_frozen or report.production_model_persisted or not report.v1_immutable or report.future_data_used
            or len(report.candidates) != len(expected) or {r["identifier"] for r in report.candidates} != expected
            or len(report.ranking) != len(expected) or set(report.ranking) != expected
            or report.best_v2_development_candidate != report.ranking[0]):
        raise BaselineReportError("Only complete, unfrozen V2 development diagnostics may be published")
    try:
        return json.dumps(asdict(report), allow_nan=False, indent=2) + "\n"
    except (TypeError, ValueError):
        raise BaselineReportError("Invalid aggregate V2 diagnostics") from None


def write_v2_report(report, path, input_roots):
    write_aggregate_report_json(v2_report_to_json(report), path, input_roots)
