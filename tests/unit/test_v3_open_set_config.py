from contextlib import contextmanager, ExitStack
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path, PureWindowsPath
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from face_recognition.application.create_v3_cv_protocol import metadata_sha256
from face_recognition.domain.models.rejection import RejectionError
from face_recognition.domain.models.v3_open_set_config import EVIDENCE_KINDS, FrozenV3Pipeline
from face_recognition.infrastructure.features.pose_landmark_normalizer import POSE_NORMALIZATION
from face_recognition.infrastructure.persistence.aggregate_report import AggregateReportError
from face_recognition.infrastructure.persistence.artifact_integrity import file_sha256
from face_recognition.infrastructure.persistence.v3_open_set_config import (
    freeze_v3_configuration_file, frozen_v3_configuration_to_json, load_v3_freeze_evidence,
    read_frozen_v3_configuration, write_frozen_v3_configuration,
)
from tests.v3_frozen_config_fakes import frozen_evidence_fixture, write_json


def resigned(config):
    return replace(config, configuration_sha256=metadata_sha256(replace(config, configuration_sha256="")))


class FrozenV3ConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory(prefix="synthetic_v3_freeze_")
        cls.folder = Path(cls.temporary.name)
        cls.paths = frozen_evidence_fixture(cls.folder)
        cls.config = load_v3_freeze_evidence(cls.paths)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @contextmanager
    def edited(self, kind, edit):
        path = self.paths[kind]
        original = path.read_bytes()
        row = json.loads(original); edit(row)
        write_json(path, row)
        try: yield
        finally: path.write_bytes(original)

    def test_exact_pose_representation_geometry_and_feature_width(self):
        c = self.config
        self.assertEqual(c.pipeline.representation, "all_xyz_pose_v2")
        self.assertEqual(c.pipeline.input_features, 1434)
        self.assertEqual(c.normalization.landmark_count, 478)
        self.assertEqual(c.normalization.first_eye, (33, 133))
        self.assertEqual(c.normalization.second_eye, (362, 263))
        self.assertEqual(c.normalization.superior_reference, (10, 109, 338))
        self.assertEqual(c.normalization.inferior_reference, (152, 148, 377))
        self.assertEqual(c.normalization.minimum_length, 1e-8)
        self.assertEqual(c.normalization.minimum_axis_sine, 1e-6)
        self.assertEqual(c.normalization_sha256, metadata_sha256(POSE_NORMALIZATION))
        self.assertEqual(len(c.implementations), 2)

    def test_exact_anova_scaler_svm_and_regularized_mahalanobis(self):
        p = self.config.pipeline
        self.assertEqual(p, FrozenV3Pipeline())
        self.assertEqual((p.selector, p.selected_features), ("SelectKBest(f_classif)", 448))
        self.assertEqual(p.scaler, "StandardScaler(copy=True, with_mean=True, with_std=True)")
        self.assertEqual(p.classifier, "SVC")
        self.assertEqual({k: dict(p.svm_parameters)[k] for k in ("kernel", "C", "gamma", "probability")},
                         dict(kernel="rbf", C=4., gamma=.001, probability=False))
        self.assertEqual(p.novelty_method, "mahalanobis_distance")
        self.assertIn("LedoitWolf", p.covariance)
        self.assertIn("Cholesky", p.covariance)
        self.assertEqual(p.covariance_ridge_relative, 1e-8)
        self.assertEqual((p.knownness_score, p.score_orientation), ("-mahalanobis_distance", "higher_is_known"))
        self.assertFalse(p.unknown_fits_model_components)

    def test_same_model_threshold_policy_no_numeric_threshold_or_deployable_allocation(self):
        p = self.config.threshold_policy
        self.assertEqual(p.threshold_transport, "none_same_model_calibration")
        self.assertTrue(p.same_fitted_state_required)
        self.assertFalse(p.post_calibration_refit_allowed)
        self.assertEqual(len(p.fitted_components), 4)
        self.assertEqual(p.calibration_level, "window_7_median_knownness")
        self.assertEqual(p.objective, "maximize_binary_balanced_accuracy")
        self.assertEqual(p.tie_breakers, ("lower_unknown_FAR", "lower_known_FRR", "higher_numeric_threshold"))
        self.assertIn("finite interior midpoints", p.enumeration)
        self.assertEqual(p.auc_eer_role, "diagnostic_only")
        self.assertIsNone(p.numeric_threshold)
        self.assertEqual(p.numeric_threshold_status, "not_calibrated")
        self.assertIn("not_defined", p.deployable_population_allocation)
        self.assertEqual(self.config.deployable_model_status, "not_fitted_by_freeze_workflow")

    def test_exact_temporal_policy(self):
        p = self.config.temporal
        self.assertEqual(p.window_size, 7)
        self.assertEqual(p.window_policy, "non_overlapping")
        self.assertEqual(p.grouping, ("person_id", "session_id", "condition"))
        self.assertEqual((p.ordering, p.incomplete_tails, p.score_aggregation), ("sample_index", "discard_and_count", "median"))
        self.assertIn("Plurality", p.identity_aggregation)
        self.assertIn("over all frames", p.identity_aggregation)
        self.assertIn("class order", p.identity_aggregation)
        self.assertFalse(p.consistency_gate)

    def test_all_five_exact_byte_digests_and_nested_immutability(self):
        self.assertEqual(tuple(r.kind for r in self.config.evidence), EVIDENCE_KINDS)
        for reference in self.config.evidence:
            self.assertEqual(reference.sha256, file_sha256(self.paths[reference.kind]))
            self.assertEqual(reference.filename, self.paths[reference.kind].name)
        with self.assertRaises(FrozenInstanceError): self.config.pipeline.selected_features = 512
        with self.assertRaises(FrozenInstanceError): self.config.evidence[0].sha256 = "0" * 64
        with self.assertRaises(TypeError): self.config.normalization.first_eye[0] = 1

    def test_changed_pose_report_bytes_changes_evidence_and_configuration_digests(self):
        path = self.paths["pose_report"]
        before = path.read_bytes()
        try:
            path.write_bytes(before + b"\n")
            changed = load_v3_freeze_evidence(self.paths)
            self.assertEqual(changed.development_evidence, self.config.development_evidence)
            self.assertNotEqual(changed.evidence[-1].sha256, self.config.evidence[-1].sha256)
            self.assertNotEqual(changed.configuration_sha256, self.config.configuration_sha256)
        finally: path.write_bytes(before)

    def test_incompatible_pipeline_or_policy_cannot_be_serialized_even_when_resigned(self):
        changes = (
            replace(self.config, pipeline=replace(self.config.pipeline, representation="all_xyz_v1")),
            replace(self.config, pipeline=replace(self.config.pipeline, input_features=1431)),
            replace(self.config, pipeline=replace(self.config.pipeline, selected_features=512)),
            replace(self.config, pipeline=replace(self.config.pipeline, novelty_method="svm_top1")),
            replace(self.config, threshold_policy=replace(self.config.threshold_policy, post_calibration_refit_allowed=True)),
            replace(self.config, threshold_policy=replace(self.config.threshold_policy, numeric_threshold=-3.)),
            replace(self.config, temporal=replace(self.config.temporal, window_size=5)),
            replace(self.config, temporal=replace(self.config.temporal, consistency_gate=True)),
            replace(self.config, status="mutable"),
        )
        for changed in changes:
            with self.subTest(status=changed.status), self.assertRaises(RejectionError):
                frozen_v3_configuration_to_json(resigned(changed))

    def test_wrong_pose_winner_pipeline_transform_or_nonfinite_metric_rejected(self):
        cases = (
            lambda r: r.update(selected_development_representation="all_xyz_v1"),
            lambda r: r.update(completion_state="incomplete"),
            lambda r: r["results"][1]["configuration"].update(selected_features=512),
            lambda r: r["results"][1]["configuration"]["svm_parameters"].__setitem__(1, ["C", 2.]),
            lambda r: r["normalization_definition"].update(minimum_length=1e-6),
            lambda r: r["results"][1]["mean_metrics"].update(binary_balanced_accuracy=float("inf")),
            lambda r: r["results"][1]["mean_metrics"].update(binary_balanced_accuracy=.1),
            lambda r: r["results"][1]["folds"][0]["evaluation"].update(no_refit_after_calibration=False),
            lambda r: r["condition_comparison"].pop(),
            lambda r: r.pop("normalization_definition_sha256"),
        )
        for edit in cases:
            # Explicit non-finite JSON is part of the malformed-input test.
            path = self.paths["pose_report"]; original = path.read_bytes()
            try:
                row = json.loads(original); edit(row)
                path.write_text(json.dumps(row), encoding="utf-8")
                with self.assertRaises(RejectionError): load_v3_freeze_evidence(self.paths)
            finally: path.write_bytes(original)

    def test_incompatible_protocol_or_cross_report_provenance_rejected(self):
        for kind, edit in (
            ("protocol", lambda r: r.update(protocol_sha256="0" * 64)),
            ("protocol", lambda r: r["specification"].update(representation="other")),
            ("novelty_report", lambda r: r["outer_results"][0].update(selected_method="svm_margin")),
            ("calibration_report", lambda r: r["selection_provenance"].update(report_sha256="0" * 64)),
            ("diagnostics_report", lambda r: r["provenance"].update(evaluation_report_sha256="0" * 64)),
            ("diagnostics_report", lambda r: r["folds"][0].update(saved_model_state_verified=False)),
            ("pose_report", lambda r: r.update(protocol_file_sha256="0" * 64)),
        ):
            with self.subTest(kind=kind), self.edited(kind, edit), self.assertRaises(RejectionError):
                load_v3_freeze_evidence(self.paths)

    def test_missing_bad_json_duplicate_keys_or_unsupported_input_paths_rejected(self):
        path = self.paths["pose_report"]; original = path.read_bytes()
        try:
            for content in ('{}', '[]', '{"schema_version":1,"schema_version":1}'):
                path.write_text(content, encoding="utf-8")
                with self.assertRaises(RejectionError): load_v3_freeze_evidence(self.paths)
        finally: path.write_bytes(original)
        for paths in ({**self.paths, "pose_report": self.folder / "missing.json"},
                      {**self.paths, "pose_report": self.folder / "payload.jsonl"},
                      {**self.paths, "pose_report": self.paths["protocol"]}):
            with self.assertRaises(RejectionError): load_v3_freeze_evidence(paths)

    def test_no_payload_access_fitting_calibration_normalization_or_experiment_replay(self):
        forbidden = (
            "face_recognition.infrastructure.persistence.v3_payload_loader.load_v3_payloads",
            "face_recognition.infrastructure.classifiers.sklearn_v3_same_model.SklearnV3SameModelRunner.fit",
            "face_recognition.infrastructure.features.pose_landmark_normalizer.PoseLandmarkNormalizer.normalize",
            "face_recognition.application.rejection_metrics.calibrate_threshold",
            "face_recognition.application.evaluate_v3_pose_representation.V3PoseRepresentationEvaluator.evaluate",
            "face_recognition.application.evaluate_v3_pose_representation.rank_representations",
        )
        original = Path.open
        def guarded(path, *args, **kwargs):
            if path.suffix in (".jsonl", ".joblib", ".npz"): raise AssertionError("No biometric/model access")
            return original(path, *args, **kwargs)
        with ExitStack() as stack:
            for name in forbidden: stack.enter_context(patch(name, side_effect=AssertionError("Freeze is metadata only")))
            stack.enter_context(patch.object(Path, "open", guarded))
            config = load_v3_freeze_evidence(self.paths)
        self.assertEqual(config, self.config)

    def test_deterministic_json_roundtrip_and_input_files_unchanged(self):
        before = {k: (p.read_bytes(), p.stat().st_mtime_ns) for k, p in self.paths.items()}
        other = load_v3_freeze_evidence(self.paths)
        self.assertEqual(frozen_v3_configuration_to_json(self.config), frozen_v3_configuration_to_json(other))
        output = self.folder / "roundtrip" / "config.json"
        freeze_v3_configuration_file(self.paths, output)
        self.assertEqual(read_frozen_v3_configuration(output), self.config)
        self.assertEqual(before, {k: (p.read_bytes(), p.stat().st_mtime_ns) for k, p in self.paths.items()})

    def test_overwrite_refused_before_evidence_reads(self):
        output = self.folder / "immutable.json"
        output.write_text("completed", encoding="utf-8")
        with patch("face_recognition.infrastructure.persistence.v3_open_set_config.load_v3_freeze_evidence") as loader:
            with self.assertRaises(AggregateReportError): freeze_v3_configuration_file(self.paths, output)
            loader.assert_not_called()
        self.assertEqual(output.read_text(encoding="utf-8"), "completed")

    def test_atomic_failure_leaves_no_partial_configuration(self):
        output = self.folder / "failed" / "config.json"
        with patch("face_recognition.infrastructure.persistence.aggregate_report.os.link", side_effect=OSError("synthetic")):
            with self.assertRaises(AggregateReportError): write_frozen_v3_configuration(self.config, output, self.paths.values())
        self.assertFalse(output.exists())
        self.assertEqual(tuple(output.parent.iterdir()), ())

    def test_manifest_tampering_or_implementation_fingerprint_mismatch_rejected(self):
        output = self.folder / "tampered.json"
        raw = json.loads(frozen_v3_configuration_to_json(self.config)); raw["evidence"][0]["sha256"] = "0" * 64
        write_json(output, raw)
        with self.assertRaises(RejectionError): read_frozen_v3_configuration(output)
        implementation = replace(self.config.implementations[0], sha256="0" * 64)
        with self.assertRaises(RejectionError):
            frozen_v3_configuration_to_json(resigned(replace(self.config, implementations=(implementation, self.config.implementations[1]))))

    def test_privacy_methodological_status_and_residual_limitations(self):
        raw = json.loads(frozen_v3_configuration_to_json(self.config))
        forbidden = {"features", "feature_values", "landmarks", "support_vectors_", "mean_", "scale_", "covariance_",
                     "per_sample_scores", "predictions", "known_ids", "unknown_ids", "selected_feature_indices"}
        def check(value):
            if isinstance(value, dict):
                self.assertFalse(set(value) & forbidden)
                for child in value.values(): check(child)
            elif isinstance(value, list):
                for child in value: check(child)
            elif isinstance(value, str): self.assertFalse(Path(value).is_absolute() or PureWindowsPath(value).is_absolute())
        check(raw)
        self.assertFalse(raw["independent_untouched_holdout_available"])
        self.assertEqual(raw["last_independent_external_holdout"], "V2")
        self.assertIn("Iterative", raw["study_status"])
        self.assertIn("pitch_up", " ".join(raw["limitations"]))
        self.assertIn("pitch_down", " ".join(raw["limitations"]))
        self.assertIn("p009", " ".join(raw["limitations"]))
        self.assertIn("relative depth", " ".join(raw["limitations"]))
