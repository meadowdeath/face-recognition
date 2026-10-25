import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from face_recognition.application.evaluate_feature_reduction import FeatureReductionEvaluator
from face_recognition.config.settings import PROJECT_ROOT
from face_recognition.infrastructure.persistence.baseline_evaluation_report import BaselineReportError
from face_recognition.infrastructure.persistence.feature_reduction_report import feature_reduction_report_to_json, write_feature_reduction_report
from tests.feature_reduction_fakes import SMALL_REDUCTIONS, RecordingReductionRunner, reduction_data


class FeatureReductionReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="synthetic_reduction_report_")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.output = self.directory / "reports/reduction.json"
        self.input = self.directory / "processed"
        self.input.mkdir()
        self.report = FeatureReductionEvaluator(RecordingReductionRunner(), SMALL_REDUCTIONS).evaluate(reduction_data())

    def test_serialization_contains_required_aggregates_dimensions_comparisons_and_ranking(self):
        write_feature_reduction_report(self.report, self.output, (self.input,))
        data = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(data["experiment"], "feature_reduction_v1")
        self.assertEqual(data["original_feature_count"], 9)
        self.assertEqual(data["source_representation"], "all_xyz_v1")
        self.assertEqual(len(data["results"]), 18)
        self.assertEqual(len(data["ranking"]), 18)
        anova = next(row for row in data["results"] if row["reduction"]["method"] == "anova")
        self.assertEqual(anova["transformations"][0]["transformation"]["selected_feature_indices"], [0, 1])
        self.assertIn("confusion_matrix", anova["evaluation"]["folds"][0]["measurements"])
        self.assertIn("macro_f1_delta", anova["comparison"])
        self.assertEqual(data["results"][0]["evaluation"]["configuration"]["classifier_parameters"]["n_neighbors"], 5)
        self.assertEqual(self.output.read_text(encoding="utf-8"), feature_reduction_report_to_json(self.report))

    def test_report_has_no_values_scores_pca_components_or_absolute_paths(self):
        data = json.loads(feature_reduction_report_to_json(self.report))

        def check(value):
            if isinstance(value, dict):
                self.assertFalse(set(value) & {"features", "landmarks", "normalized_landmarks", "scores", "components", "X", "y", "samples"})
                for child in value.values():
                    check(child)
            elif isinstance(value, list):
                for child in value:
                    check(child)
            elif isinstance(value, str):
                self.assertFalse(Path(value).is_absolute())
                self.assertNotIn(str(self.directory), value)

        check(data)

    def test_existing_completed_report_is_never_overwritten(self):
        write_feature_reduction_report(self.report, self.output, (self.input,))
        before = self.output.read_bytes()
        with self.assertRaisesRegex(BaselineReportError, "already exists"):
            write_feature_reduction_report(self.report, self.output, (self.input,))
        self.assertTrue(before == self.output.read_bytes())

    def test_report_rejects_repository_input_tree_and_failed_publication(self):
        for path in (PROJECT_ROOT / "reports/reduction.json", self.input / "reduction.json"):
            with self.assertRaises(BaselineReportError):
                write_feature_reduction_report(self.report, path, (self.input,))
        with patch("face_recognition.infrastructure.persistence.baseline_evaluation_report.os.link", side_effect=OSError("private")):
            with self.assertRaises(BaselineReportError):
                write_feature_reduction_report(self.report, self.output, (self.input,))
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.output.parent.iterdir()), [])
