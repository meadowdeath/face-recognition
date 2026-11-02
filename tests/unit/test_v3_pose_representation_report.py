from dataclasses import replace
import json
from pathlib import Path, PureWindowsPath
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from face_recognition.application.evaluate_v3_pose_representation import V3PoseRepresentationEvaluator
from face_recognition.domain.models.rejection import RejectionError
from face_recognition.infrastructure.features.pose_landmark_normalizer import POSE_NORMALIZATION
from face_recognition.infrastructure.persistence.aggregate_report import AggregateReportError
from face_recognition.infrastructure.persistence.v3_pose_representation import pose_representation_report_to_json, write_pose_representation_report
from tests.pose_representation_fakes import paired_pose_data
from tests.v3_mahalanobis_fakes import RecordingSameModelRunner


class V3PoseReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory(prefix="synthetic_pose_report_")
        cls.folder = Path(cls.temporary.name)
        baseline, candidate, catalog, cls.processed, _ = paired_pose_data(cls.folder)
        cls.report = V3PoseRepresentationEvaluator(RecordingSameModelRunner).evaluate(
            baseline, candidate, "a" * 64, catalog, POSE_NORMALIZATION)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_report_contains_both_configurations_all_metrics_pitch_down_and_provenance(self):
        raw = json.loads(pose_representation_report_to_json(self.report))
        self.assertEqual([r["representation"] for r in raw["results"]], ["all_xyz_v1", "all_xyz_pose_v2"])
        self.assertEqual(raw["threshold_transport"], "none_same_model_calibration")
        self.assertEqual(raw["protocol_sha256"], self.report.protocol_sha256)
        self.assertIn("pitch_down", [r["condition"] for r in raw["condition_comparison"]])
        for result in raw["results"]:
            self.assertIn("macro_precision", result["mean_closed_set"])
            self.assertIn("macro_recall", result["mean_closed_set"])
            self.assertIn("known_misidentification_rate", result["mean_metrics"])
            self.assertIn("binary_macro_f1", result["mean_metrics"])
            self.assertEqual(len(result["folds"]), 4)
            self.assertEqual(len(result["diagnostics"]["known_session_severity"]), 4)
            self.assertEqual(len(result["diagnostics"]["unknown_identity_severity"]), 8)
            self.assertEqual(result["configuration"]["window_size"], 7)

    def test_aggregate_report_has_no_vectors_individual_scores_or_absolute_paths(self):
        raw = json.loads(pose_representation_report_to_json(self.report))
        def check(value):
            if isinstance(value, dict):
                self.assertFalse(set(value) & {"features", "landmarks", "support_vectors_", "predicted", "decisions", "sample_index", "window_scores", "_centers", "_factors"})
                for child in value.values(): check(child)
            elif isinstance(value, list):
                for child in value: check(child)
            elif isinstance(value, str):
                self.assertFalse(Path(value).is_absolute() or PureWindowsPath(value).is_absolute())
        check(raw)
        self.assertIn("Iterative", raw["methodology"])
        self.assertIn("V2 remains the last independent external holdout", raw["methodology"])

    def test_atomic_publication_refuses_overwrite(self):
        output = self.folder / "published" / "pose.json"
        write_pose_representation_report(self.report, output, (self.processed,))
        before = output.read_bytes(), output.stat().st_mtime_ns
        with self.assertRaises(AggregateReportError): write_pose_representation_report(self.report, output, (self.processed,))
        self.assertEqual(before, (output.read_bytes(), output.stat().st_mtime_ns))

    def test_failed_atomic_publication_leaves_no_partial_report(self):
        output = self.folder / "failed" / "pose.json"
        with patch("face_recognition.infrastructure.persistence.aggregate_report.os.link", side_effect=OSError("synthetic failure")):
            with self.assertRaises(AggregateReportError): write_pose_representation_report(self.report, output, (self.processed,))
        self.assertFalse(output.exists())
        self.assertEqual(tuple(output.parent.iterdir()), ())

    def test_incomplete_report_wrong_winner_or_missing_pitch_down_rejected(self):
        for report in (replace(self.report, completion_state="failed"),
                       replace(self.report, selected_development_representation="all_xyz_pose_v2"),
                       replace(self.report, condition_comparison=tuple(r for r in self.report.condition_comparison if r.condition != "pitch_down")),
                       replace(self.report, pose_distance_frr_deltas=tuple(r for r in self.report.pose_distance_frr_deltas if r[0] != "pitch_down"))):
            with self.assertRaises(RejectionError): pose_representation_report_to_json(report)

    def test_changed_model_or_refit_claim_cannot_be_published(self):
        result = self.report.results[0]
        fold = result.folds[0]
        for changed in (replace(fold.evaluation, fit_calls=2), replace(fold.evaluation, no_refit_after_calibration=False),
                        replace(fold.evaluation, model_sha256_after_test="0" * 64)):
            invalid = replace(result, folds=(replace(fold, evaluation=changed), *result.folds[1:]))
            with self.assertRaises(RejectionError):
                pose_representation_report_to_json(replace(self.report, results=(invalid, self.report.results[1])))

    def test_changed_classifier_or_fold_roles_cannot_be_published(self):
        from face_recognition.application.create_v3_cv_protocol import metadata_sha256
        result = self.report.results[1]
        config = replace(result.configuration, selected_features=512)
        invalid = replace(result, configuration=config, configuration_sha256=metadata_sha256(config))
        with self.assertRaises(RejectionError):
            pose_representation_report_to_json(replace(self.report, results=(self.report.results[0], invalid)))
        fold = result.folds[0]
        invalid = replace(result, folds=(replace(fold, evaluation=replace(fold.evaluation,
            roles=replace(fold.evaluation.roles, known_test_session="different"))), *result.folds[1:]))
        with self.assertRaises(RejectionError):
            pose_representation_report_to_json(replace(self.report, results=(self.report.results[0], invalid)))
