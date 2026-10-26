from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, replace
from hashlib import sha256
import unittest

from face_recognition.application.freeze_open_set_config import freeze_development_configuration
from face_recognition.domain.models.open_set_config import ConfigurationFreezeError, DevelopmentReportProvenance
from face_recognition.infrastructure.persistence.open_set_config_repository import frozen_configuration_to_json
from tests.open_set_config_fakes import synthetic_development_report, synthetic_report_bytes


class OpenSetConfigTests(unittest.TestCase):
    def setUp(self):
        self.report = synthetic_development_report()
        self.digest = sha256(synthetic_report_bytes()).hexdigest()

    def freeze(self):
        return freeze_development_configuration(self.report, self.digest)

    def test_expected_winner_and_all_fixed_choices(self):
        config = self.freeze()
        arch = config.architecture
        self.assertEqual((arch.source_representation, arch.original_feature_count, arch.feature_selection_method,
                          arch.feature_selection_score, arch.selected_features), ("all_xyz_v1", 1434, "SelectKBest", "f_classif", 448))
        self.assertEqual(arch.scaler_type, "StandardScaler")
        self.assertEqual(dict(arch.scaler_parameters), {"copy": True, "with_mean": True, "with_std": True})
        self.assertEqual(arch.classifier_type, "SVC")
        self.assertEqual(dict(arch.svm_parameters), {"kernel": "rbf", "C": 4., "gamma": .001, "probability": False,
                                                   "cache_size": 128, "random_state": 42})
        self.assertEqual((arch.novelty_method, arch.novelty_neighbor_count, arch.novelty_metric), ("mean_5nn", 5, "euclidean"))
        self.assertEqual((arch.temporal_window_size, arch.novelty_temporal_aggregation, arch.identity_temporal_aggregation), (7, "median", "majority_vote"))
        self.assertEqual(len(arch.identity_tie_breaking), 3)
        self.assertIn("ALL window frames", arch.identity_tie_breaking[1])
        self.assertIn("class order", arch.identity_tie_breaking[2])
        self.assertIsNone(config.final_numeric_threshold)
        self.assertIn("not_numerically_calibrated", config.final_threshold_status)
        self.assertEqual((config.status, config.completion_state), ("development_configuration_frozen", "frozen"))
        self.assertFalse(config.final_holdouts_evaluated or config.final_model_fitted)

    def test_evidence_is_copied_from_source_not_historical_constants(self):
        before = deepcopy(self.report)
        config = self.freeze()
        selected = next(row for row in self.report["results"] if row["configuration_id"] == "mean_5nn__window_7")
        self.assertEqual(config.development_evidence.mean_roc_auc, selected["mean_roc_auc"])
        self.assertEqual(config.development_evidence.mean_eer, selected["mean_eer"])
        self.assertEqual(asdict(config.development_evidence.mean_metrics), selected["mean_metrics"])
        self.assertEqual(config.development_evidence.known_window_observations, selected["total_known_counts"]["windows"])
        self.assertEqual(self.report, before)
        self.report["results"][-1]["mean_roc_auc"] = .9
        with self.assertRaisesRegex(ConfigurationFreezeError, "disagrees with folds"):
            self.freeze()

    def test_model_mismatches_rejected(self):
        for field, value in (("k", 512), ("C", 2.), ("gamma", .01), ("kernel", "linear"), ("probability", True)):
            with self.subTest(field=field):
                report = deepcopy(self.report)
                report["model"][field] = value
                with self.assertRaises(ConfigurationFreezeError):
                    freeze_development_configuration(report, self.digest)
        for field, value in (("source_representation", "other_v1"), ("original_feature_count", 512)):
            report = deepcopy(self.report)
            report[field] = value
            with self.assertRaises(ConfigurationFreezeError):
                freeze_development_configuration(report, self.digest)

    def test_wrong_selected_method_or_window_rejected(self):
        for candidate in ("nearest_1__window_7", "mean_5nn__window_5"):
            report = deepcopy(self.report)
            report["best_temporal_distance_novelty_development_candidate"] = candidate
            with self.assertRaisesRegex(ConfigurationFreezeError, "selected candidate"):
                freeze_development_configuration(report, self.digest)
        for field, value in (("score_family", "nearest_1"), ("window_size", 5)):
            report = deepcopy(self.report)
            report["results"][-1][field] = value
            with self.assertRaises(ConfigurationFreezeError):
                freeze_development_configuration(report, self.digest)

    def test_incomplete_report_folds_or_inconsistent_ranking_rejected(self):
        reports = []
        for key in ("completion_state", "configuration", "results", "threshold_policy"):
            report = deepcopy(self.report)
            del report[key]
            reports.append(report)
        report = deepcopy(self.report)
        report["results"][-1]["folds"].pop()
        reports.append(report)
        report = deepcopy(self.report)
        del report["results"][-1]["folds"][0]["analysis"]["calibration"]["metrics"]
        reports.append(report)
        report = deepcopy(self.report)
        report["ranking"] = list(reversed(report["ranking"]))
        reports.append(report)
        for report in reports:
            with self.assertRaises(ConfigurationFreezeError):
                freeze_development_configuration(report, self.digest)

    def test_changed_scaling_voting_aggregation_or_protected_source_rejected(self):
        for key in ("confidence_policy", "identity_policy", "window_policy", "threshold_policy", "distance_space"):
            report = deepcopy(self.report)
            report[key] = "different policy"
            with self.assertRaises(ConfigurationFreezeError):
                freeze_development_configuration(report, self.digest)
        report = deepcopy(self.report)
        report["configuration"]["scaler_parameters"]["with_std"] = False
        with self.assertRaises(ConfigurationFreezeError):
            freeze_development_configuration(report, self.digest)
        for key in ("known_test_payloads_accessed", "unknown_holdout_payloads_accessed"):
            report = deepcopy(self.report)
            report[key] = True
            with self.assertRaises(ConfigurationFreezeError):
                freeze_development_configuration(report, self.digest)

    def test_nonfinite_or_invalid_evidence_rejected_without_values(self):
        for value in (float("nan"), float("inf"), -float("inf"), True, -1., 2.):
            report = deepcopy(self.report)
            report["results"][-1]["mean_roc_auc"] = value
            with self.assertRaises(ConfigurationFreezeError):
                freeze_development_configuration(report, self.digest)
        report = deepcopy(self.report)
        report["results"][-1]["folds"][0]["analysis"]["calibration"]["metrics"]["known_rejection_rate"] = .5
        with self.assertRaises(ConfigurationFreezeError):
            freeze_development_configuration(report, self.digest)

    def test_threshold_policy_allowed_sources_self_exclusion_and_window_level_search(self):
        config = self.freeze()
        policy = config.final_threshold_calibration_policy
        self.assertEqual(policy.allowed_data_sources, ("known_train_pool", "unknown_calibration"))
        self.assertEqual(policy.prohibited_data_sources, ("known_test", "unknown_holdout"))
        self.assertEqual(policy.fitting_partition, "known_train_pool")
        self.assertTrue(policy.self_exclusion)
        self.assertEqual(policy.self_exclusion_key, ("person_id", "session_id", "condition", "sample_index"))
        self.assertEqual((policy.minimum_other_reference_samples, policy.minimum_total_known_reference_samples), (5, 6))
        self.assertIn("distinct duplicate-valued samples remain eligible", policy.self_exclusion_rule)
        self.assertEqual(policy.selection_rule, ("higher binary balanced accuracy", "lower FAR", "lower FRR", "higher threshold"))
        self.assertIn("WINDOW level", policy.ordered_steps[7])
        self.assertIn("BEFORE", policy.ordered_steps[-1])
        self.assertIn("Do not copy", policy.fold_threshold_transfer)
        self.assertFalse(config.development_window_policy.overlapping)
        self.assertIn("discard trailing", config.development_window_policy.incomplete_windows)

    def test_metric_definitions_and_frozen_immutability(self):
        config = self.freeze()
        definitions = dict(config.metric_definitions.definitions)
        for name in ("ROC_AUC", "EER", "FAR", "FRR", "known_correct_identification", "known_misidentification",
                     "open_set_exact_accuracy", "binary_balanced_accuracy", "binary_macro_F1"):
            self.assertIn(name, definitions)
        with self.assertRaises(FrozenInstanceError):
            config.architecture.selected_features = 512
        with self.assertRaises(ValueError):
            replace(config, final_numeric_threshold=-1.)
        with self.assertRaises(FrozenInstanceError):
            config.status = "unfrozen"
        with self.assertRaises(ConfigurationFreezeError):
            DevelopmentReportProvenance("invalid digest")
        self.assertEqual(frozen_configuration_to_json(config), frozen_configuration_to_json(self.freeze()))
