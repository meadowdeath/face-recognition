from pathlib import Path
import json
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from face_recognition.application.evaluate_baseline_models import BaselineIdentityEvaluator
from face_recognition.config.settings import PROJECT_ROOT
from face_recognition.infrastructure.classifiers.sklearn_baselines import SklearnBaselineRunner
from face_recognition.infrastructure.persistence.baseline_evaluation_report import (
    BaselineReportError, baseline_report_to_json, validate_baseline_report_target, write_baseline_report,
)
from tests.baseline_fakes import development_data


class BaselineReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = BaselineIdentityEvaluator(SklearnBaselineRunner()).evaluate(development_data())

    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="synthetic_baseline_report_")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.processed = self.directory / "processed"
        self.processed.mkdir()
        self.split = self.directory / "split.json"
        self.split.write_text("synthetic split input placeholder", encoding="utf-8")
        self.target = self.directory / "reports" / "baseline.json"
        self.inputs = (self.processed, self.split)

    def test_report_serializes_configurations_folds_aggregates_and_confusion_matrices(self):
        write_baseline_report(self.report, self.target, self.inputs)
        data = json.loads(self.target.read_text(encoding="utf-8"))
        self.assertEqual((data["completion_state"], data["representation"], data["feature_count"]), ("completed", "all_xyz_v1", 9))
        self.assertEqual(data["development_samples"], 24)
        self.assertEqual(data["development_partition"], "known_train_pool")
        self.assertEqual(data["label_order"], ["known_alpha", "known_beta"])
        self.assertEqual(data["classifiers"][0]["configuration"]["classifier_parameters"]["n_neighbors"], 5)
        self.assertIsNone(data["classifiers"][2]["configuration"]["scaler_parameters"])
        self.assertEqual(data["classifiers"][0]["folds"][0]["measurements"]["confusion_matrix"], [[6, 0], [0, 6]])
        self.assertEqual(data["classifiers"][0]["mean_metrics"]["accuracy"], 1)
        self.assertIn("scikit_learn", data["versions"])
        self.assertEqual(self.target.read_text(encoding="utf-8"), baseline_report_to_json(self.report))

    def test_report_contains_no_vectors_coordinates_or_absolute_input_paths(self):
        text = baseline_report_to_json(self.report)

        def check(value):
            if isinstance(value, dict):
                self.assertFalse(set(value) & {"features", "landmarks", "normalized_landmarks", "X", "y", "samples"})
                for child in value.values():
                    check(child)
            elif isinstance(value, list):
                for child in value:
                    check(child)
            elif isinstance(value, str):
                self.assertFalse(Path(value).is_absolute())
                self.assertNotIn(str(self.directory), value)

        check(json.loads(text))

    def test_completed_report_cannot_be_overwritten(self):
        write_baseline_report(self.report, self.target, self.inputs)
        before = self.target.read_bytes()
        with self.assertRaisesRegex(BaselineReportError, "already exists"):
            write_baseline_report(self.report, self.target, self.inputs)
        self.assertTrue(self.target.read_bytes() == before, "Existing report remains unchanged")

    def test_report_target_must_be_external_and_outside_input_trees(self):
        for path in (PROJECT_ROOT / "reports/baseline.json", self.processed / "baseline.json", self.processed, self.split):
            with self.subTest(case=path.name), self.assertRaises(BaselineReportError):
                validate_baseline_report_target(path, self.inputs)
        self.assertFalse(self.target.exists())

    def test_failed_atomic_publication_leaves_no_partial_final_or_temporary_file(self):
        marker = "SYNTHETIC_PRIVATE_PAYLOAD"
        with patch("face_recognition.infrastructure.persistence.baseline_evaluation_report.os.link", side_effect=OSError(marker)):
            with self.assertRaises(BaselineReportError) as caught:
                write_baseline_report(self.report, self.target, self.inputs)
        self.assertNotIn(marker, str(caught.exception))
        self.assertFalse(self.target.exists())
        self.assertEqual(list(self.target.parent.iterdir()), [])

    def test_report_publication_does_not_rewrite_inputs(self):
        original = self.split.read_bytes(), self.split.stat().st_mtime_ns
        write_baseline_report(self.report, self.target, self.inputs)
        self.assertTrue((self.split.read_bytes(), self.split.stat().st_mtime_ns) == original)
        self.assertEqual(list(self.processed.iterdir()), [])
