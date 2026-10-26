from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from face_recognition.application.evaluate_temporal_open_set import TemporalOpenSetEvaluator
from face_recognition.config.settings import PROJECT_ROOT
from face_recognition.infrastructure.classifiers.sklearn_temporal_open_set import SklearnTemporalOpenSetRunner
from face_recognition.infrastructure.persistence.baseline_evaluation_report import BaselineReportError
from face_recognition.infrastructure.persistence.temporal_open_set_report import temporal_report_to_json, write_temporal_report
from tests.unknown_rejection_fakes import rejection_fixture, load_rejection_fixture


class TemporalOpenSetReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory(prefix="synthetic_temporal_report_")
        cls.directory = Path(cls.temporary.name)
        cls.processed, source, partition, cls.source_path, cls.partition_path = rejection_fixture(cls.directory, samples=8)
        known, unknown = load_rejection_fixture(cls.processed, source, partition)
        cls.report = TemporalOpenSetEvaluator(SklearnTemporalOpenSetRunner()).evaluate(known, unknown)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_complete_serialization_contains_counts_diagnostics_comparisons_and_rank(self):
        output = self.directory / "reports/complete.json"
        write_temporal_report(self.report, output, (self.processed, self.source_path, self.partition_path))
        data = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(data["experiment"], "temporal_open_set_v1")
        self.assertEqual(data["model"], {"k": 448, "C": 4., "gamma": .001, "kernel": "rbf", "probability": False})
        self.assertEqual(data["window_sizes"], [1, 3, 5, 7])
        self.assertEqual(len(data["results"]), 8)
        self.assertEqual(len(data["ranking"]), 8)
        self.assertEqual(data["best_temporal_open_set_development_candidate"], data["ranking"][0])
        for row in data["results"]:
            self.assertEqual(len(row["folds"]), 2)
            self.assertIn("vs_window1", row)
            self.assertIn("discarded_samples", row["total_known_counts"])
            self.assertIn("conditions", row)
            self.assertIn("mean_vote_rates", row)
            for fold in row["folds"]:
                self.assertEqual(len(fold["analysis"]["far_operating_points"]), 3)
                self.assertIn("roc_auc", fold["analysis"])
                self.assertIn("equal_error_point", fold["analysis"])
        self.assertFalse(data["known_test_payloads_accessed"])
        self.assertFalse(data["unknown_holdout_payloads_accessed"])
        self.assertEqual(output.read_text(encoding="utf-8"), temporal_report_to_json(self.report))

    def test_no_biometric_sequences_vectors_participant_condition_details_or_absolute_paths(self):
        data = json.loads(temporal_report_to_json(self.report))
        forbidden = {"features", "landmarks", "normalized_landmarks", "centroids", "scales", "scores", "decisions",
                     "support_vectors_", "known_confidence", "unknown_confidence", "predicted", "indices", "vote_sequences", "X", "y"}

        def check(value):
            if isinstance(value, dict):
                self.assertFalse(set(value) & forbidden)
                for child in value.values():
                    check(child)
            elif isinstance(value, list):
                for child in value:
                    check(child)
            elif isinstance(value, str):
                self.assertFalse(Path(value).is_absolute())
                self.assertNotIn(str(self.directory), value)
        check(data)
        for row in data["results"]:
            for condition in row["conditions"]:
                self.assertNotIn("person_id", condition)
                self.assertNotIn("session_id", condition)

    def test_existing_completed_report_and_unsafe_paths_refused(self):
        output = self.directory / "reports/existing.json"
        write_temporal_report(self.report, output, (self.processed,))
        before = output.read_bytes()
        with self.assertRaisesRegex(BaselineReportError, "already exists"):
            write_temporal_report(self.report, output, (self.processed,))
        self.assertTrue(before == output.read_bytes())
        for path in (PROJECT_ROOT / "reports/temporal.json", self.processed / "temporal.json"):
            with self.assertRaises(BaselineReportError):
                write_temporal_report(self.report, path, (self.processed,))

    def test_incomplete_or_unprotected_reports_cannot_be_published(self):
        for report in (replace(self.report, results=self.report.results[:-1]), replace(self.report, ranking=()),
                       replace(self.report, known_test_payloads_accessed=True),
                       replace(self.report, unknown_holdout_payloads_accessed=True),
                       replace(self.report, results=(replace(self.report.results[0], vs_window1=None), *self.report.results[1:]))):
            with self.assertRaises(BaselineReportError):
                temporal_report_to_json(report)

    def test_atomic_failure_does_not_publish_partial_report(self):
        output = self.directory / "failed/temporal.json"
        with patch("face_recognition.infrastructure.persistence.baseline_evaluation_report.os.link", side_effect=OSError("private")):
            with self.assertRaises(BaselineReportError):
                write_temporal_report(self.report, output, (self.processed,))
        self.assertFalse(output.exists())
        self.assertEqual(list(output.parent.iterdir()), [])
