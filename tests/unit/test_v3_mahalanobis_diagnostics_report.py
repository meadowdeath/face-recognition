from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path, PureWindowsPath
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from face_recognition.application.analyze_v3_mahalanobis_diagnostics import V3MahalanobisDiagnosticAnalyzer
from face_recognition.domain.models.rejection import RejectionError
from face_recognition.infrastructure.persistence.aggregate_report import AggregateReportError
from face_recognition.infrastructure.persistence.artifact_integrity import file_sha256
from face_recognition.infrastructure.persistence.v3_mahalanobis_diagnostics_report import (
    mahalanobis_diagnostics_to_json, read_mahalanobis_evaluation_report, write_mahalanobis_diagnostics,
)
from tests.v3_mahalanobis_diagnostics_fakes import DiagnosticReplayRunner, diagnostic_fixture


class MahalanobisDiagnosticReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory(prefix="synthetic_v3_diagnostic_reports_")
        cls.folder = Path(cls.temporary.name)
        cls.data, cls.processed, cls.protocol_path, cls.source_path, cls.source, cls.provenance = diagnostic_fixture(cls.folder)
        cls.report = V3MahalanobisDiagnosticAnalyzer(DiagnosticReplayRunner()).analyze(cls.data, cls.source, cls.provenance)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def read(self, path=None, protocol_sha=None):
        return read_mahalanobis_evaluation_report(path or self.source_path, self.data.protocol,
            protocol_sha or file_sha256(self.protocol_path))

    def test_complete_source_roundtrip_exact_digest_and_no_initial_report_access(self):
        # Embedded selection provenance suffices; its original file is not an
        # input and cannot accidentally trigger method selection again.
        selection = self.folder / "selection.json"
        before = selection.read_bytes()
        selection.unlink()
        try:
            source, digest = self.read()
            self.assertEqual(source, self.source)
            self.assertEqual(digest, file_sha256(self.source_path))
        finally:
            selection.write_bytes(before)

    def test_source_metadata_configuration_provenance_status_and_summaries_fail_closed(self):
        raw = json.loads(self.source_path.read_text(encoding="utf-8"))
        changes = (
            ("protocol_sha256", "0" * 64), ("protocol_file_sha256", "0" * 64),
            ("catalog_sha256", "0" * 64), ("processing_manifest_sha256", "0" * 64),
            ("completion_state", "incomplete"), ("configuration_sha256", "0" * 64),
            ("experiment", "different_study"), ("threshold_transport", "direct_transfer"),
            ("mean_roc_auc", .123), ("source_samples", 1), ("identity_count", 99),
        )
        path = self.folder / "invalid.json"
        for field, value in changes:
            changed = deepcopy(raw); changed[field] = value
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.subTest(field=field), self.assertRaises(RejectionError): self.read(path)
        for mutate in (
            lambda r: r.pop("versions"),
            lambda r: r["configuration"].update(selected_features=512),
            lambda r: r["selection_provenance"].update(payload_sha256="0" * 64),
            lambda r: r["outer_results"][0].update(threshold=float("nan")),
            lambda r: r["outer_results"][0].update(fit_calls=2),
            lambda r: r["outer_results"][0].update(model_sha256_after_test="0" * 64),
            lambda r: r["outer_results"][0]["roles"].update(known_test_session="future"),
            lambda r: r["outer_results"][0]["known_test_counts"].update(used_samples=0),
            lambda r: r["outer_results"][0]["diagnostics"]["known_distribution"].update(count=0),
            lambda r: r["outer_results"].pop(),
        ):
            changed = deepcopy(raw); mutate(changed)
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaises(RejectionError): self.read(path)

    def test_different_protocol_bytes_or_source_changed_while_reading_rejected(self):
        with self.assertRaises(RejectionError): self.read(protocol_sha="0" * 64)
        with patch("face_recognition.infrastructure.persistence.v3_mahalanobis_diagnostics_report.file_sha256",
                   side_effect=[file_sha256(self.source_path), "0" * 64]):
            with self.assertRaises(RejectionError): self.read()

    def test_duplicate_json_keys_and_nonfinite_saved_evidence_rejected(self):
        path = self.folder / "malformed.json"
        for content in ('{"schema_version":1,"schema_version":1}', '{"score":Infinity}', '[]'):
            path.write_text(content, encoding="utf-8")
            with self.assertRaises(RejectionError): self.read(path)

    def test_atomic_aggregate_publication_privacy_and_immutable_source_files(self):
        output = self.folder / "reports" / "diagnostics.json"
        before = [(p.read_bytes(), p.stat().st_mtime_ns) for p in (self.source_path, self.protocol_path)]
        write_mahalanobis_diagnostics(self.report, output, (self.processed, self.source_path, self.protocol_path))
        raw = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(raw["provenance"]["evaluation_report_sha256"], file_sha256(self.source_path))
        self.assertEqual(raw["provenance"]["evaluation_report_reference"], self.source_path.name)
        self.assertEqual(raw["method"], "mahalanobis_distance")
        self.assertEqual(len(raw["folds"]), 4)
        self.assertEqual(len(raw["aggregate"]["known_condition_by_session"]), 4)
        self.assertEqual(before, [(p.read_bytes(), p.stat().st_mtime_ns) for p in (self.source_path, self.protocol_path)])
        self.assertEqual(tuple(output.parent.iterdir()), (output,))
        forbidden = {"features", "landmarks", "predicted", "predictions", "decisions", "sample_index", "keys",
                     "frame_scores", "window_scores", "individual_scores", "_centers", "_factors", "support_vectors_"}
        def check(value):
            if isinstance(value, dict):
                self.assertFalse(set(value) & forbidden)
                for child in value.values(): check(child)
            elif isinstance(value, list):
                for child in value: check(child)
            elif isinstance(value, str):
                self.assertFalse(Path(value).is_absolute() or PureWindowsPath(value).is_absolute())
        check(raw)

    def test_existing_output_or_source_cannot_be_overwritten(self):
        output = self.folder / "refused.json"
        output.write_text("immutable", encoding="utf-8")
        for target in (output, self.source_path, self.protocol_path):
            before = target.read_bytes(), target.stat().st_mtime_ns
            with self.assertRaises(AggregateReportError):
                write_mahalanobis_diagnostics(self.report, target, (self.processed, self.source_path, self.protocol_path))
            self.assertEqual(before, (target.read_bytes(), target.stat().st_mtime_ns))

    def test_failed_atomic_publication_leaves_no_partial_output(self):
        output = self.folder / "failed" / "diagnostics.json"
        with patch("face_recognition.infrastructure.persistence.aggregate_report.os.link", side_effect=OSError("synthetic")):
            with self.assertRaises(AggregateReportError):
                write_mahalanobis_diagnostics(self.report, output, (self.processed, self.source_path))
        self.assertFalse(output.exists())
        self.assertEqual(tuple(output.parent.iterdir()), ())

    def test_unverified_or_incomplete_diagnostics_cannot_be_published(self):
        for report in (replace(self.report, completion_state="failed"), replace(self.report, folds=self.report.folds[:3]),
                       replace(self.report, folds=(replace(self.report.folds[0], saved_model_state_verified=False), *self.report.folds[1:]))):
            with self.assertRaises(RejectionError): mahalanobis_diagnostics_to_json(report)

    def test_diagnostics_bytes_deterministic_and_input_digest_changes_with_source_bytes(self):
        self.assertEqual(mahalanobis_diagnostics_to_json(self.report), mahalanobis_diagnostics_to_json(self.report))
        path = self.folder / "different_format.json"
        path.write_bytes(self.source_path.read_bytes() + b"\n")
        source, digest = self.read(path)
        self.assertEqual(source, self.source)
        self.assertNotEqual(digest, self.provenance.evaluation_report_sha256)
