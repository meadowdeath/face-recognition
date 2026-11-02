"""Post-hoc condition/session diagnostics; previous evidence is read-only."""

import argparse
from pathlib import Path
import sys

from face_recognition.application.analyze_v3_mahalanobis_diagnostics import V3MahalanobisDiagnosticAnalyzer
from face_recognition.domain.models.v3_mahalanobis_diagnostics import MahalanobisDiagnosticProvenance
from face_recognition.infrastructure.classifiers.sklearn_v3_same_model import SklearnV3SameModelRunner
from face_recognition.infrastructure.persistence.aggregate_report import validate_aggregate_report_target
from face_recognition.infrastructure.persistence.artifact_integrity import file_sha256
from face_recognition.infrastructure.persistence.v3_cv_protocol_repository import read_v3_catalog, read_v3_cv_protocol
from face_recognition.infrastructure.persistence.v3_mahalanobis_diagnostics_report import (
    read_mahalanobis_evaluation_report, write_mahalanobis_diagnostics,
)
from face_recognition.infrastructure.persistence.v3_payload_loader import load_v3_payloads


def main(argv=None):
    parser = argparse.ArgumentParser(description="Post-hoc V3 diagnostics only; no selection or calibration decisions changed")
    for name in ("processed-root", "protocol", "evaluation-report", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        inputs = (args.processed_root, args.protocol, args.evaluation_report)
        target = validate_aggregate_report_target(args.report, inputs)
        protocol_sha = file_sha256(args.protocol)
        protocol = read_v3_cv_protocol(args.protocol)
        evidence, evidence_sha = read_mahalanobis_evaluation_report(args.evaluation_report, protocol, protocol_sha)
        print("V3 MAHALANOBIS POST-HOC DIAGNOSTICS — deterministic evidence replay, not a new evaluation", flush=True)
        data = load_v3_payloads(args.processed_root, protocol)
        provenance = MahalanobisDiagnosticProvenance(args.protocol.name, protocol.protocol_sha256, protocol_sha,
            args.evaluation_report.name, evidence_sha, evidence.selection_provenance.report_sha256, evidence.catalog_sha256,
            evidence.processing_manifest_sha256, evidence.payload_sha256, evidence.configuration_sha256)
        result = V3MahalanobisDiagnosticAnalyzer(SklearnV3SameModelRunner(),
                   progress=lambda message: print(message, flush=True)).analyze(data, evidence, provenance)
        if (file_sha256(args.protocol) != protocol_sha or file_sha256(args.evaluation_report) != evidence_sha
                or read_v3_catalog(args.processed_root) != protocol.catalog
                or any((args.processed_root / name).stat().st_mtime_ns != stamp for name, stamp in data.source_stamps)):
            raise ValueError
        write_mahalanobis_diagnostics(result, target, inputs)
    except KeyboardInterrupt:
        print("Diagnostics interrupted; check for a completed diagnostics report before retrying.", file=sys.stderr)
        return 1
    except Exception:
        print("Diagnostics failed: invalid provenance/payload, non-reproducible saved evidence or existing output. No partial diagnostics report published.", file=sys.stderr)
        return 1
    print("DIAGNOSTICS ONLY — no model selection/calibration decisions changed")
    for line in result.interpretation:
        print(line)
    print("Detailed condition/session/identity, condition-by-session and contribution tables are in the aggregate JSON report.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
