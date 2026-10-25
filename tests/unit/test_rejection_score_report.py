from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from face_recognition.application.evaluate_rejection_scores import RejectionScoreEvaluator
from face_recognition.config.settings import PROJECT_ROOT
from face_recognition.infrastructure.classifiers.sklearn_rejection_scores import SklearnRejectionScoreRunner
from face_recognition.infrastructure.persistence.baseline_evaluation_report import BaselineReportError
from face_recognition.infrastructure.persistence.rejection_score_report import rejection_score_report_to_json, write_rejection_score_report
from tests.unknown_rejection_fakes import rejection_fixture, load_rejection_fixture


class RejectionScoreReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory(prefix="synthetic_score_report_")
        cls.directory = Path(cls.temporary.name)
        cls.processed, source, partition, cls.source_path, cls.partition_path = rejection_fixture(cls.directory)
        known, unknown = load_rejection_fixture(cls.processed, source, partition)
        cls.report = RejectionScoreEvaluator(SklearnRejectionScoreRunner()).evaluate(known, unknown)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_complete_serialization_preserves_all_families_operating_points_and_rank(self):
        output = self.directory / "reports/complete.json"
        write_rejection_score_report(self.report, output, (self.processed, self.source_path, self.partition_path))
        data = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(data["experiment"], "unknown_rejection_scores_v1")
        self.assertEqual(len(data["results"]), 8)
        self.assertEqual(len(data["ranking"]), 8)
        self.assertEqual(data["best_rejection_score_development_candidate"], data["ranking"][0])
        first = data["results"][0]
        self.assertIn("roc_auc", first["folds"][0]["analysis"])
        self.assertIn("equal_error_point", first["folds"][0]["analysis"])
        self.assertEqual(len(first["folds"][0]["analysis"]["far_operating_points"]), 3)
        self.assertIn("pooled_known_distribution", first)
        self.assertIn("vs_margin", first)
        self.assertFalse(data["known_test_payloads_accessed"])
        self.assertFalse(data["unknown_holdout_payloads_accessed"])
        self.assertEqual(output.read_text(encoding="utf-8"), rejection_score_report_to_json(self.report))

    def test_no_vectors_centroids_scales_scores_or_paths_are_serialized(self):
        data = json.loads(rejection_score_report_to_json(self.report))

        def check(value):
            if isinstance(value, dict):
                self.assertFalse(set(value) & {"features", "landmarks", "normalized_landmarks", "scores", "centroids", "scales",
                                               "decision_function", "support_vectors_", "known_confidence", "unknown_confidence", "X", "y"})
                for child in value.values():
                    check(child)
            elif isinstance(value, list):
                for child in value:
                    check(child)
            elif isinstance(value, str):
                self.assertFalse(Path(value).is_absolute())
                self.assertNotIn(str(self.directory), value)
        check(data)

    def test_existing_completed_report_and_unsafe_targets_refused(self):
        output = self.directory / "reports/existing.json"
        write_rejection_score_report(self.report, output, (self.processed,))
        before = output.read_bytes()
        with self.assertRaisesRegex(BaselineReportError, "already exists"):
            write_rejection_score_report(self.report, output, (self.processed,))
        self.assertTrue(before == output.read_bytes())
        for path in (PROJECT_ROOT / "reports/scores.json", self.processed / "scores.json"):
            with self.assertRaises(BaselineReportError):
                write_rejection_score_report(self.report, path, (self.processed,))

    def test_incomplete_and_unprotected_reports_rejected(self):
        for report in (replace(self.report, results=self.report.results[:-1]), replace(self.report, ranking=()),
                       replace(self.report, known_test_payloads_accessed=True),
                       replace(self.report, unknown_holdout_payloads_accessed=True)):
            with self.assertRaises(BaselineReportError):
                rejection_score_report_to_json(report)

    def test_atomic_publication_failure_leaves_no_partial_report(self):
        output = self.directory / "failed/score.json"
        with patch("face_recognition.infrastructure.persistence.baseline_evaluation_report.os.link", side_effect=OSError("private")):
            with self.assertRaises(BaselineReportError):
                write_rejection_score_report(self.report, output, (self.processed,))
        self.assertFalse(output.exists())
        self.assertEqual(list(output.parent.iterdir()), [])
