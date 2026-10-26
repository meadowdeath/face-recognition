from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from face_recognition.application.evaluate_final_open_set import FinalOpenSetEvaluator
from face_recognition.application.final_evaluation_data import FinalEvaluationError
from face_recognition.infrastructure.classifiers.sklearn_final_open_set import SklearnFinalOpenSetRunner
from face_recognition.infrastructure.persistence.baseline_evaluation_report import BaselineReportError
from face_recognition.infrastructure.persistence.final_evaluation_report import final_evaluation_report_to_json, write_final_evaluation_report
from tests.final_evaluation_fakes import final_fixture, load_final_fixture


class FinalEvaluationReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory(prefix="synthetic_final_report_")
        cls.directory = Path(cls.temporary.name)
        cls.bundle = final_fixture(cls.directory)
        data, model, context = load_final_fixture(cls.bundle)
        cls.report = FinalOpenSetEvaluator(SklearnFinalOpenSetRunner(model)).evaluate(data, context)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_complete_report_sections_threshold_provenance_and_comparison(self):
        output = self.directory / "reports/complete.json"
        write_final_evaluation_report(self.report, output, (self.bundle[0][0], self.bundle[2]), self.bundle[2])
        text = output.read_text(encoding="utf-8")
        self.assertEqual(text, final_evaluation_report_to_json(self.report))
        data = json.loads(text)
        self.assertEqual(data["status"], "final_holdout_evaluation_completed")
        self.assertEqual(data["context"]["threshold"], self.report.context.threshold)
        self.assertEqual(data["context"]["model_manifest_sha256"], dict(self.report.context.model_fingerprints)["manifest.json"])
        for key in ("closed_set_known_frames", "frame_novelty_diagnostics", "operational_metrics", "temporal_novelty_diagnostics",
                    "conditions", "known_vote_stability", "development_candidate", "final_holdout", "calibration_warning", "no_retuning"):
            self.assertIn(key, data)
        self.assertTrue(data["model_artifacts_unchanged"])
        self.assertFalse(data["fitting_performed"] or data["threshold_calibration_performed"])
        self.assertTrue(data["known_test_payloads_accessed"] and data["unknown_holdout_payloads_accessed"])

    def test_no_vectors_individual_scores_predictions_or_absolute_paths(self):
        data = json.loads(final_evaluation_report_to_json(self.report))
        forbidden = {"features", "landmarks", "normalized_landmarks", "support_vectors_", "reference_matrix", "reference_vectors",
                     "nearest_labels", "individual_distances", "scores", "known_confidence", "unknown_confidence", "predicted",
                     "sample_index", "vote_sequences", "X", "y"}

        def inspect(value):
            if isinstance(value, dict):
                self.assertFalse(set(value) & forbidden)
                for child in value.values():
                    inspect(child)
            elif isinstance(value, list):
                for child in value:
                    inspect(child)
            elif isinstance(value, str):
                self.assertFalse(Path(value).is_absolute())
                self.assertNotIn(str(self.directory), value)
        inspect(data)
        for condition in data["conditions"]:
            self.assertNotIn("person_id", condition)
            self.assertNotIn("session_id", condition)
        self.assertEqual(data["closed_set_known_frames"]["label_order"], list(self.report.closed_set_known_frames.label_order))

    def test_completed_report_not_overwritten(self):
        output = self.directory / "reports/no_overwrite.json"
        write_final_evaluation_report(self.report, output, (self.bundle[2],), self.bundle[2])
        before = output.read_bytes()
        with self.assertRaisesRegex(BaselineReportError, "already exists"):
            write_final_evaluation_report(self.report, output, (self.bundle[2],), self.bundle[2])
        self.assertTrue(before == output.read_bytes())

    def test_changed_model_rejects_report_publication(self):
        file = self.bundle[2] / "threshold.json"
        before = file.read_bytes()
        try:
            file.write_bytes(before + b"\n")
            with self.assertRaisesRegex(FinalEvaluationError, "changed"):
                write_final_evaluation_report(self.report, self.directory / "reports/refused.json", (), self.bundle[2])
        finally:
            file.write_bytes(before)
        self.assertFalse((self.directory / "reports/refused.json").exists())

    def test_incomplete_or_refitted_report_rejected(self):
        for report in (replace(self.report, fitting_performed=True), replace(self.report, threshold_calibration_performed=True),
                       replace(self.report, model_artifacts_unchanged=False), replace(self.report, no_retuning=""),
                       replace(self.report, known_test_payloads_accessed=False)):
            with self.assertRaises(FinalEvaluationError):
                final_evaluation_report_to_json(report)
