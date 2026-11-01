from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from face_recognition.application.tune_anova_svm import AnovaSvmTuner
from face_recognition.config.settings import PROJECT_ROOT
from face_recognition.infrastructure.classifiers.sklearn_anova_svm import SklearnAnovaSvmRunner
from face_recognition.infrastructure.persistence.anova_svm_tuning_report import anova_svm_report_to_json, write_anova_svm_report
from face_recognition.infrastructure.persistence.baseline_evaluation_report import BaselineReportError
from tests.feature_reduction_fakes import reduction_data


class AnovaSvmTuningReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = AnovaSvmTuner(SklearnAnovaSvmRunner()).evaluate(reduction_data(feature_count=768))

    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="synthetic_tuning_report_")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.input = self.directory / "processed"
        self.input.mkdir()
        self.output = self.directory / "reports/tuning.json"

    def test_all_configurations_ranking_diagnostics_and_selected_candidate_are_serialized(self):
        write_anova_svm_report(self.report, self.output, (self.input,))
        data = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(data["experiment"], "anova_svm_tuning_v1")
        self.assertEqual(data["source_representation"], "all_xyz_v1")
        self.assertEqual(len(data["results"]), 60)
        self.assertEqual(len(data["ranking"]), 60)
        self.assertEqual(len(data["selection_stability"]), 5)
        self.assertEqual(data["selected_development_candidate"]["candidate_id"], data["ranking"][0]["candidate_id"])
        self.assertEqual(data["results"][0]["evaluation"]["configuration"]["classifier_parameters"]["kernel"], "rbf")
        self.assertEqual(len(data["results"][0]["evaluation"]["folds"]), 2)
        self.assertIn("confusion_matrix", data["results"][0]["evaluation"]["folds"][0]["measurements"])
        self.assertIn("counts", data["results"][0]["support_vectors"][0])
        self.assertIn("vs_full_svm_baseline", data["results"][0])
        self.assertIn("vs_anova512_reference", data["results"][0])
        self.assertEqual(self.output.read_text(encoding="utf-8"), anova_svm_report_to_json(self.report))

    def test_report_contains_no_biometric_values_or_absolute_paths(self):
        data = json.loads(anova_svm_report_to_json(self.report))

        def check(value):
            if isinstance(value, dict):
                self.assertFalse(set(value) & {"features", "landmarks", "normalized_landmarks", "scores", "components",
                                               "support_vectors_", "support_vector_values", "X", "y", "samples"})
                for child in value.values():
                    check(child)
            elif isinstance(value, list):
                for child in value:
                    check(child)
            elif isinstance(value, str):
                self.assertFalse(Path(value).is_absolute())
                self.assertNotIn(str(self.directory), value)

        check(data)

    def test_existing_completed_report_refuses_overwrite(self):
        write_anova_svm_report(self.report, self.output, (self.input,))
        before = self.output.read_bytes()
        with self.assertRaisesRegex(BaselineReportError, "already exists"):
            write_anova_svm_report(self.report, self.output, (self.input,))
        self.assertTrue(before == self.output.read_bytes())

    def test_repository_input_overlap_and_incomplete_report_are_rejected(self):
        for path in (PROJECT_ROOT / "reports/tuning.json", self.input / "tuning.json"):
            with self.assertRaises(BaselineReportError):
                write_anova_svm_report(self.report, path, (self.input,))
        for report in (replace(self.report, results=self.report.results[:-1]),
                       replace(self.report, ranking=self.report.ranking[:-1]),
                       replace(self.report, configurations_evaluated=59),
                       replace(self.report, ranking=tuple(reversed(self.report.ranking))),
                       replace(self.report, completion_state="failed")):
            with self.assertRaises(BaselineReportError):
                write_anova_svm_report(report, self.output, (self.input,))
        self.assertFalse(self.output.exists())

    def test_failed_atomic_publication_leaves_no_partial_files(self):
        with patch("face_recognition.infrastructure.persistence.baseline_evaluation_report.os.link", side_effect=OSError("private")):
            with self.assertRaises(BaselineReportError) as caught:
                write_anova_svm_report(self.report, self.output, (self.input,))
        self.assertNotIn("private", str(caught.exception))
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.output.parent.iterdir()), [])
