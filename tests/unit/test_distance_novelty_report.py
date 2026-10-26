from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from face_recognition.application.evaluate_distance_novelty import DistanceNoveltyEvaluator
from face_recognition.config.settings import PROJECT_ROOT
from face_recognition.infrastructure.classifiers.sklearn_distance_novelty import SklearnDistanceNoveltyRunner
from face_recognition.infrastructure.persistence.baseline_evaluation_report import BaselineReportError
from face_recognition.infrastructure.persistence.distance_novelty_report import distance_novelty_report_to_json, write_distance_novelty_report
from tests.unknown_rejection_fakes import rejection_fixture, load_rejection_fixture


class DistanceNoveltyReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory(prefix="synthetic_distance_report_")
        cls.directory = Path(cls.temporary.name)
        cls.processed, source, partition, cls.source_path, cls.partition_path = rejection_fixture(cls.directory)
        known, unknown = load_rejection_fixture(cls.processed, source, partition)
        cls.report = DistanceNoveltyEvaluator(SklearnDistanceNoveltyRunner()).evaluate(known, unknown)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_complete_report_and_deterministic_serialization(self):
        output = self.directory / "reports/complete.json"
        write_distance_novelty_report(self.report, output, (self.processed, self.source_path, self.partition_path))
        data = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(data["experiment"], "distance_novelty_v1")
        self.assertEqual(data["model"], {"k": 448, "C": 4., "gamma": .001, "kernel": "rbf", "probability": False})
        self.assertEqual([r["neighbor_count"] for r in data["novelty_results"]], [1, 3, 5])
        self.assertEqual([r["score_family"] for r in data["reference_results"]], ["svm_margin", "centroid_distance"])
        self.assertEqual(data["best_distance_based_novelty_development_candidate"], data["ranking"][0])
        self.assertEqual(len(data["neighbor_diagnostics"]), 2)
        for row in data["novelty_results"]:
            self.assertEqual(len(row["folds"]), 2)
            self.assertEqual(len(row["vs_references"]), 2)
            self.assertIn("p10", row["known_distance_distribution"])
            self.assertIn("p90", row["unknown_distance_distribution"])
            for fold in row["folds"]:
                self.assertEqual(len(fold["analysis"]["far_operating_points"]), 3)
                self.assertIn("equal_error_point", fold["analysis"])
        self.assertFalse(data["known_test_payloads_accessed"])
        self.assertFalse(data["unknown_holdout_payloads_accessed"])
        self.assertEqual(output.read_text(encoding="utf-8"), distance_novelty_report_to_json(self.report))

    def test_no_features_reference_arrays_individual_distances_neighbor_ids_or_absolute_paths(self):
        data = json.loads(distance_novelty_report_to_json(self.report))
        forbidden = {"features", "landmarks", "normalized_landmarks", "centroids", "scales", "scores", "distances",
                     "reference", "reference_matrix", "nearest_labels", "nearest_indices", "nearest_neighbors", "sample_index",
                     "person_id", "session_id", "support_vectors_", "known_confidence", "unknown_confidence", "X", "y"}

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

    def test_completed_report_overwrite_and_unsafe_targets_refused(self):
        output = self.directory / "reports/existing.json"
        write_distance_novelty_report(self.report, output, (self.processed,))
        before = output.read_bytes()
        with self.assertRaisesRegex(BaselineReportError, "already exists"):
            write_distance_novelty_report(self.report, output, (self.processed,))
        self.assertEqual(output.read_bytes(), before)
        for target in (PROJECT_ROOT / "reports/novelty.json", self.processed / "novelty.json"):
            with self.assertRaises(BaselineReportError):
                write_distance_novelty_report(self.report, target, (self.processed,))

    def test_incomplete_or_unprotected_report_rejected(self):
        for report in (replace(self.report, ranking=()), replace(self.report, novelty_results=self.report.novelty_results[:-1]),
                       replace(self.report, known_test_payloads_accessed=True), replace(self.report, unknown_holdout_payloads_accessed=True),
                       replace(self.report, reference_results=()), replace(self.report, neighbor_diagnostics=()),
                       replace(self.report, novelty_results=(replace(self.report.novelty_results[0], vs_references=()), *self.report.novelty_results[1:]))):
            with self.assertRaises(BaselineReportError):
                distance_novelty_report_to_json(report)

    def test_atomic_failure_does_not_publish_partial_report(self):
        output = self.directory / "failed/novelty.json"
        with patch("face_recognition.infrastructure.persistence.baseline_evaluation_report.os.link", side_effect=OSError("private")):
            with self.assertRaises(BaselineReportError):
                write_distance_novelty_report(self.report, output, (self.processed,))
        self.assertFalse(output.exists())
        self.assertEqual(list(output.parent.iterdir()), [])
