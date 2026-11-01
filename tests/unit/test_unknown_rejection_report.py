from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from face_recognition.application.calibrate_unknown_rejection import UnknownRejectionCalibrator
from face_recognition.config.settings import PROJECT_ROOT
from face_recognition.infrastructure.classifiers.sklearn_unknown_rejection import SklearnUnknownRejectionRunner
from face_recognition.infrastructure.persistence.baseline_evaluation_report import BaselineReportError
from face_recognition.infrastructure.persistence.unknown_rejection_repository import rejection_report_to_json, write_rejection_report
from tests.unknown_rejection_fakes import load_rejection_fixture, rejection_fixture


class UnknownRejectionReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory(prefix="synthetic_rejection_report_")
        cls.directory = Path(cls.temporary.name)
        cls.processed, cls.source, cls.partition, cls.source_path, cls.partition_path = rejection_fixture(cls.directory)
        known, unknown = load_rejection_fixture(cls.processed, cls.source, cls.partition)
        cls.report = UnknownRejectionCalibrator(SklearnUnknownRejectionRunner()).evaluate(known, unknown)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_complete_report_serializes_aggregate_calibration_and_median_metrics(self):
        output = self.directory / "reports/complete.json"
        write_rejection_report(self.report, output, (self.processed, self.source_path, self.partition_path))
        data = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(data["experiment"], "unknown_rejection_calibration_v1")
        self.assertFalse(data["protected_payloads_accessed"])
        self.assertEqual(len(data["results"]), 2)
        self.assertEqual(len(data["ranking"]), 2)
        self.assertIn("development_rejection_threshold", data["results"][0])
        self.assertIn("at_development_threshold", data["results"][0]["folds"][0])
        self.assertIn("mean_closed_set_metrics", data["results"][0])
        self.assertEqual(data["results"][0]["configuration"]["classifier_parameters"]["C"], 4)
        self.assertEqual(output.read_text(encoding="utf-8"), rejection_report_to_json(self.report))

    def test_no_features_scores_confidences_coordinates_or_absolute_paths_are_persisted(self):
        def check(value):
            if isinstance(value, dict):
                self.assertFalse(set(value) & {"features", "landmarks", "normalized_landmarks", "scores", "known_confidence",
                                               "unknown_confidence", "decision_function", "support_vectors_", "X", "y", "samples"})
                for child in value.values():
                    check(child)
            elif isinstance(value, list):
                for child in value:
                    check(child)
            elif isinstance(value, str):
                self.assertFalse(Path(value).is_absolute())
                self.assertNotIn(str(self.directory), value)
        # Partition counts legitimately use aggregate 'samples'; validate them
        # separately, then inspect every candidate/metric subtree for row data.
        data = json.loads(rejection_report_to_json(self.report))
        partition = data.pop("unknown_partition")
        self.assertNotIn(str(self.directory), json.dumps(partition))
        self.assertFalse(set(partition) & {"features", "landmarks", "scores"})
        check(data)

    def test_no_overwrite_or_repository_input_tree_outputs(self):
        output = self.directory / "reports/existing.json"
        inputs = (self.processed, self.source_path, self.partition_path)
        write_rejection_report(self.report, output, inputs)
        before = output.read_bytes()
        with self.assertRaisesRegex(BaselineReportError, "already exists"):
            write_rejection_report(self.report, output, inputs)
        self.assertTrue(before == output.read_bytes())
        for path in (PROJECT_ROOT / "reports/rejection.json", self.processed / "rejection.json", self.partition_path):
            with self.assertRaises(BaselineReportError):
                write_rejection_report(self.report, path, inputs)

    def test_incomplete_or_falsely_claimed_protection_reports_rejected(self):
        for report in (replace(self.report, results=self.report.results[:1]),
                       replace(self.report, protected_payloads_accessed=True), replace(self.report, ranking=())):
            with self.assertRaises(BaselineReportError):
                rejection_report_to_json(report)

    def test_atomic_publication_failure_does_not_leave_partial_report(self):
        output = self.directory / "failed/report.json"
        with patch("face_recognition.infrastructure.persistence.baseline_evaluation_report.os.link", side_effect=OSError("private")):
            with self.assertRaises(BaselineReportError):
                write_rejection_report(self.report, output, (self.processed,))
        self.assertFalse(output.exists())
        self.assertEqual(list(output.parent.iterdir()), [])
