from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from face_recognition.application.evaluate_v3_mahalanobis_calibration import V3MahalanobisCalibrationEvaluator
from face_recognition.domain.models.rejection import RejectionError
from face_recognition.infrastructure.persistence.aggregate_report import AggregateReportError
from face_recognition.infrastructure.persistence.v3_mahalanobis_calibration_report import (
    mahalanobis_calibration_report_to_json, read_mahalanobis_selection_report, write_mahalanobis_calibration_report,
)
from tests.v3_mahalanobis_fakes import RecordingSameModelRunner, selection_report_data, write_selection_report
from tests.v3_novelty_fakes import novelty_data


class V3MahalanobisReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory(prefix="synthetic_v3_calibration_report_")
        cls.folder = Path(cls.temporary.name)
        cls.data, cls.processed, _ = novelty_data(cls.folder)
        cls.selection_path = write_selection_report(cls.data, cls.folder / "selection.json")
        cls.selection = read_mahalanobis_selection_report(cls.selection_path, cls.data.protocol)
        cls.report = V3MahalanobisCalibrationEvaluator(RecordingSameModelRunner()).evaluate(cls.data, "a" * 64, cls.selection)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_invalid_protocol_configuration_selection_or_completion_fails(self):
        cases = []
        for field, value in (("protocol_sha256", "0" * 64), ("catalog_sha256", "0" * 64),
                             ("configuration_sha256", "0" * 64), ("completion_state", "incomplete")):
            source = selection_report_data(self.data); source[field] = value; cases.append(source)
        source = selection_report_data(self.data)
        source["configuration"]["selected_features"] = 512
        cases.append(source)
        source = selection_report_data(self.data)
        source["outer_results"][0]["selected_method"] = "svm_top1"
        cases.append(source)
        source = selection_report_data(self.data)
        del source["payload_sha256"]
        cases.append(source)
        path = self.folder / "invalid_selection.json"
        for source in cases:
            path.write_text(json.dumps(source), encoding="utf-8")
            with self.subTest(source=source.get("completion_state")), self.assertRaises(RejectionError):
                read_mahalanobis_selection_report(path, self.data.protocol)

    def test_source_dataset_payload_mismatch_fails_before_fitting(self):
        runner = RecordingSameModelRunner()
        with self.assertRaises(RejectionError):
            V3MahalanobisCalibrationEvaluator(runner).evaluate(self.data, "a" * 64, replace(self.selection, payload_sha256="0" * 64))
        self.assertEqual(runner.fit_count, 0)

    def test_publication_is_aggregate_only_and_records_methodology_and_integrity(self):
        output = self.folder / "published" / "study.json"
        before = self.selection_path.read_bytes(), self.selection_path.stat().st_mtime_ns
        write_mahalanobis_calibration_report(self.report, output, (self.processed, self.selection_path))
        raw = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(raw["threshold_transport"], "none_same_model_calibration")
        self.assertEqual(len(raw["outer_results"]), 4)
        self.assertIn("refinement", raw["methodology"])
        self.assertIn("no additional untouched holdout", raw["limitation"])
        self.assertEqual(raw["selection_provenance"]["transport_warning_folds"], 4)
        self.assertEqual(before, (self.selection_path.read_bytes(), self.selection_path.stat().st_mtime_ns))
        self.assertEqual(tuple(output.parent.iterdir()), (output,))
        def check(value):
            if isinstance(value, dict):
                self.assertFalse(set(value) & {"features", "landmarks", "predicted", "decisions", "_centers", "_factors", "support_vectors_", "individual_scores"})
                for child in value.values(): check(child)
            elif isinstance(value, list):
                for child in value: check(child)
            elif isinstance(value, str): self.assertFalse(Path(value).is_absolute())
        check(raw)

    def test_completed_output_overwrite_is_refused(self):
        output = self.folder / "immutable" / "study.json"
        write_mahalanobis_calibration_report(self.report, output, (self.processed, self.selection_path))
        before = output.read_bytes(), output.stat().st_mtime_ns
        with self.assertRaises(AggregateReportError):
            write_mahalanobis_calibration_report(self.report, output, (self.processed, self.selection_path))
        self.assertEqual(before, (output.read_bytes(), output.stat().st_mtime_ns))

    def test_failed_atomic_publication_leaves_no_report_or_staging(self):
        output = self.folder / "failed" / "study.json"
        with patch("face_recognition.infrastructure.persistence.aggregate_report.os.link", side_effect=OSError("synthetic failure")):
            with self.assertRaises(AggregateReportError):
                write_mahalanobis_calibration_report(self.report, output, (self.processed, self.selection_path))
        self.assertFalse(output.exists())
        self.assertEqual(tuple(output.parent.iterdir()), ())

    def test_changed_model_or_refit_claim_cannot_be_serialized_as_completed(self):
        row = self.report.outer_results[0]
        for invalid in (replace(row, fit_calls=2), replace(row, no_refit_after_calibration=False),
                        replace(row, same_model_identity_verified=False), replace(row, model_sha256_after_test="0" * 64)):
            with self.assertRaises(RejectionError):
                mahalanobis_calibration_report_to_json(replace(self.report, outer_results=(invalid, *self.report.outer_results[1:])))
