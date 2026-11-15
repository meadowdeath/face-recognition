"""Frozen operational contract validation; no research freezing workflow."""
from dataclasses import replace
import json
from pathlib import Path, PureWindowsPath
from tempfile import TemporaryDirectory
import unittest
from face_recognition.application.recognition_config import metadata_sha256
from face_recognition.domain.models.rejection import RejectionError
from face_recognition.domain.models.v3_open_set_config import FrozenV3Pipeline
from face_recognition.infrastructure.features.pose_landmark_normalizer import POSE_NORMALIZATION
from face_recognition.infrastructure.persistence.recognition_config import frozen_configuration_to_json, read_frozen_configuration
from tests.recognition_fakes import deployable_fixture


def resigned(config):
    return replace(config, configuration_sha256=metadata_sha256(replace(config, configuration_sha256="")))


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


class RecognitionConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory(prefix="synthetic_config_")
        cls.folder = Path(cls.temporary.name)
        _, cls.config, _, _, cls.config_path, *_ = deployable_fixture(cls.folder)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

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
                frozen_configuration_to_json(resigned(changed))

    def test_manifest_tampering_or_implementation_fingerprint_mismatch_rejected(self):
        output = self.folder / "tampered.json"
        raw = json.loads(frozen_configuration_to_json(self.config)); raw["evidence"][0]["sha256"] = "0" * 64
        write_json(output, raw)
        with self.assertRaises(RejectionError): read_frozen_configuration(output)
        implementation = replace(self.config.implementations[0], sha256="0" * 64)
        with self.assertRaises(RejectionError):
            frozen_configuration_to_json(resigned(replace(self.config, implementations=(implementation, self.config.implementations[1]))))

    def test_privacy_methodological_status_and_residual_limitations(self):
        raw = json.loads(frozen_configuration_to_json(self.config))
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

    def test_deterministic_roundtrip_and_malformed_json_rejected(self):
        self.assertEqual(read_frozen_configuration(self.config_path), self.config)
        self.assertEqual(frozen_configuration_to_json(self.config), frozen_configuration_to_json(self.config))
        bad = self.folder / "bad.json"
        for content in ('{}', '[]', '{"schema_version":1,"schema_version":1}', 'NaN'):
            bad.write_text(content, encoding="utf-8")
            with self.assertRaises(RejectionError): read_frozen_configuration(bad)
