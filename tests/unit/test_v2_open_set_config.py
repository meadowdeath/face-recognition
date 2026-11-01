from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, replace
from hashlib import sha256
import json
import unittest

from face_recognition.application.freeze_v2_open_set_config import freeze_v2_configuration
from face_recognition.domain.models.open_set_config import ConfigurationFreezeError
from face_recognition.domain.models.v2_open_set_config import V2ReportProvenance, V2_FROZEN_CANDIDATE
from face_recognition.infrastructure.persistence.v2_open_set_config_repository import frozen_v2_configuration_to_json
from tests.v2_open_set_config_fakes import synthetic_v2_report


class V2OpenSetConfigTests(unittest.TestCase):
    def setUp(self):
        self.report = synthetic_v2_report()
        self.digest = sha256(json.dumps(self.report).encode()).hexdigest()

    def freeze(self, report=None):
        return freeze_v2_configuration(self.report if report is None else report, self.digest)

    def selected(self, report=None):
        return next(r for r in (self.report if report is None else report)["candidates"] if r["identifier"] == V2_FROZEN_CANDIDATE)

    def test_expected_winner_identity_architecture_and_state(self):
        c = self.freeze()
        a = c.architecture
        self.assertEqual(c.provenance.selected_candidate, V2_FROZEN_CANDIDATE)
        self.assertEqual((a.source_representation, a.original_feature_count, a.feature_selection_method,
                          a.feature_selection_score, a.selected_features), ("all_xyz_v1", 1434, "SelectKBest", "f_classif", 448))
        self.assertEqual(a.scaler_type, "StandardScaler")
        self.assertEqual(dict(a.scaler_parameters), {"copy": True, "with_mean": True, "with_std": True})
        self.assertEqual(a.classifier_type, "SVC")
        self.assertEqual(dict(a.svm_parameters), {"kernel": "rbf", "C": 4., "gamma": .001, "probability": False,
                                                "cache_size": 128, "random_state": 42})
        self.assertEqual((c.status, c.completion_state), ("development_configuration_frozen", "frozen"))
        self.assertFalse(c.final_model_fitted or c.v2_final_evaluation_performed or c.biometric_payloads_accessed)
        self.assertTrue(c.v1_immutable)
        self.assertIsNone(c.final_numeric_threshold)

    def test_score_temporal_policy_and_ties(self):
        a = self.freeze().architecture
        self.assertEqual((a.rejection_strategy, a.rejection_score_family, a.rejection_threshold_type), ("svm_top1_global", "svm_top1", "global"))
        self.assertIn("higher", a.score_orientation)
        self.assertIn("Maximum SVM", a.frame_score_definition)
        self.assertEqual((a.temporal_window_size, a.rejection_temporal_aggregation, a.identity_temporal_aggregation), (7, "median", "majority_vote"))
        self.assertFalse(a.identity_consistency_gate)
        self.assertEqual(len(a.identity_tie_breaking), 3)
        self.assertIn("ALL window frames", a.identity_tie_breaking[1])
        self.assertIn("class order", a.identity_tie_breaking[2])
        w = self.freeze().development_window_policy
        self.assertEqual((w.order_by, w.group_by, w.overlapping), ("sample_index", ("person_id", "session_id", "condition"), False))
        self.assertIn("discard", w.incomplete_windows)

    def test_oof_threshold_policy_precise_and_not_resubstitution(self):
        c = self.freeze()
        p = c.final_threshold_calibration_policy
        self.assertFalse(p.same_training_pool_resubstitution or p.copy_fold_threshold)
        self.assertIn("every known window occurs once", p.known_calibration)
        self.assertIn("ALL current", p.unknown_calibration)
        self.assertIn("EACH", p.unknown_calibration)
        self.assertIn("three model observations", p.unknown_observation_accounting)
        self.assertIn("ALL original", p.candidate_selection_history)
        self.assertEqual(p.selection_rule, ("higher binary balanced accuracy", "lower FAR", "lower FRR", "higher threshold"))
        self.assertEqual(p.diagnostic_far_targets, (.2, .1, .05))
        self.assertIn("Diagnostics only", p.diagnostic_far_policy)
        self.assertIn("direct OOF", p.score_scale_alignment)
        self.assertIn("BEFORE", p.transfer_limitation)
        self.assertIn("DEVELOPMENT", p.transfer_limitation)
        self.assertIn(">=", p.decision_rule)
        for f in c.known_cross_session_folds:
            self.assertNotIn(f.calibration_session, f.fit_sessions)
            self.assertEqual(len(f.fit_sessions), 2)

    def test_future_data_boundary_and_metric_definitions(self):
        c = self.freeze()
        future = c.future_final_test
        self.assertTrue(future.configuration_freeze_before_collection and future.fitted_model_and_threshold_freeze_before_collection)
        self.assertFalse(future.future_data_accessed or future.final_evaluation_performed)
        self.assertIn("session_04", future.known_test)
        self.assertIn("never used", future.unknown_test)
        self.assertIn("not an algorithm requirement", future.unknown_count_guidance)
        self.assertIn("known frames BEFORE", dict(c.metric_definitions.definitions)["closed_set_accuracy"])
        self.assertIn("unknown", dict(c.metric_definitions.definitions)["FAR"].lower())

    def test_report_not_mutated_evidence_and_provenance_come_from_report(self):
        before = deepcopy(self.report)
        config = self.freeze()
        selected = self.selected()
        e = config.development_evidence
        self.assertEqual(e.mean_roc_auc, selected["mean_roc_auc"])
        self.assertEqual(e.mean_eer, selected["mean_eer"])
        self.assertEqual(asdict(e.metrics), selected["metrics"])
        self.assertEqual(e.known_source_samples, 84)
        self.assertEqual(e.unknown_source_samples, 28)
        self.assertEqual(e.known_unique_windows, 12)
        self.assertEqual(e.unknown_three_model_window_observations, 12)
        self.assertEqual(config.provenance.sha256, self.digest)
        self.assertEqual(config.provenance.v1_final_report_sha256, "a" * 64)
        self.assertEqual(config.provenance.original_identity_split_sha256, "b" * 64)
        self.assertEqual(self.report, before)
        self.assertIn("not final", e.scope)

    def test_wrong_winner_or_ranking_rejected(self):
        for change in ("winner", "ranking"):
            r = deepcopy(self.report)
            if change == "winner":
                r["best_v2_development_candidate"] = "mean_5nn_global__window_7__normal"
            else:
                r["ranking"] = list(reversed(r["ranking"]))
            with self.assertRaises(ConfigurationFreezeError):
                self.freeze(r)

    def test_wrong_anova_scaler_svm_representation_rejected(self):
        for name, value in (("selector", "PCA"), ("selected_features", 512), ("scaler", "none"), ("classifier", "KNN"),
                            ("C", 2.), ("gamma", .01), ("kernel", "linear"), ("probability", True)):
            with self.subTest(name=name):
                r = deepcopy(self.report)
                for pair in r["identity_model"]:
                    if pair[0] == name:
                        pair[1] = value
                with self.assertRaises(ConfigurationFreezeError):
                    self.freeze(r)
        for key, value in (("source_representation", "other"), ("original_feature_count", 448)):
            r = deepcopy(self.report)
            r[key] = value
            with self.assertRaises(ConfigurationFreezeError):
                self.freeze(r)

    def test_wrong_rejection_threshold_type_window_or_gate_rejected(self):
        for name, value in (("score_family", "mean_5nn"), ("strategy", "svm_margin_global"),
                            ("per_identity", True), ("window", 5), ("strict_majority_gate", True)):
            r = deepcopy(self.report)
            self.selected(r)["configuration"][name] = value
            with self.assertRaises(ConfigurationFreezeError):
                self.freeze(r)

    def test_policy_and_flags_cannot_be_overridden(self):
        for name, value in (("threshold_policy", "train on all known and calibrate on resubstitution"),
                            ("temporal_policy", "mean scores"), ("v2_frozen", True), ("future_data_used", True),
                            ("production_model_persisted", True), ("v1_immutable", False), ("schema_version", True)):
            r = deepcopy(self.report)
            r[name] = value
            with self.assertRaises(ConfigurationFreezeError):
                self.freeze(r)

    def test_incomplete_report_rejected(self):
        for key in ("v1_baseline", "folds", "closed_set_frame_oof", "score_distributions", "unknown_data_role"):
            r = deepcopy(self.report)
            del r[key]
            with self.assertRaises(ConfigurationFreezeError):
                self.freeze(r)
        for key in ("candidates", "score_distributions"):
            r = deepcopy(self.report)
            r[key].pop()
            with self.assertRaises(ConfigurationFreezeError):
                self.freeze(r)
        r = deepcopy(self.report)
        self.selected(r)["cells"].pop()
        with self.assertRaises(ConfigurationFreezeError):
            self.freeze(r)

    def test_nonfinite_and_inconsistent_metrics_rejected(self):
        for value in (float("nan"), float("inf"), float("-inf"), -.1, 1.1):
            r = deepcopy(self.report)
            self.selected(r)["mean_roc_auc"] = value
            with self.assertRaises(ConfigurationFreezeError):
                self.freeze(r)
        r = deepcopy(self.report)
        self.selected(r)["cells"][0]["roc_auc"] = .1
        with self.assertRaises(ConfigurationFreezeError):
            self.freeze(r)
        r = deepcopy(self.report)
        self.selected(r)["metrics"]["known_rejection_rate"] = .2
        with self.assertRaises(ConfigurationFreezeError):
            self.freeze(r)

    def test_confusion_and_fold_coverage_inconsistency_rejected(self):
        r = deepcopy(self.report)
        self.selected(r)["open_set_confusion"]["observation_counts"][0][0] += 1
        with self.assertRaises(ConfigurationFreezeError):
            self.freeze(r)
        r = deepcopy(self.report)
        r["folds"][0]["fit_sessions"].append("visit_c")
        with self.assertRaises(ConfigurationFreezeError):
            self.freeze(r)
        r = deepcopy(self.report)
        self.selected(r)["cells"][0]["held_unknown_rotation"] = 2
        with self.assertRaises(ConfigurationFreezeError):
            self.freeze(r)

    def test_missing_or_malformed_provenance_rejected(self):
        for key in ("report_sha256", "identity_split_sha256"):
            r = deepcopy(self.report)
            r["v1_baseline"][key] = "invalid"
            with self.assertRaises(ConfigurationFreezeError):
                self.freeze(r)
        with self.assertRaises(ConfigurationFreezeError):
            freeze_v2_configuration(self.report, "bad")

    def test_incomplete_score_and_vote_evidence_rejected(self):
        r = deepcopy(self.report)
        del r["score_distributions"][0]["known_correct"]["statistics"]["p90"]
        with self.assertRaises(ConfigurationFreezeError):
            self.freeze(r)
        r = deepcopy(self.report)
        self.selected(r)["votes"][0]["known"]["unanimous_windows"] += 1
        with self.assertRaises(ConfigurationFreezeError):
            self.freeze(r)

    def test_privacy_and_only_configuration_is_serialized(self):
        payload = frozen_v2_configuration_to_json(self.freeze())
        parsed = json.loads(payload)
        for key in ('"features"', '"landmarks"', '"support_vectors"', '"per_sample_scores"',
                    '"scaler_state"', '"mean_5nn"', "synthetic_a", "synthetic_b", "C:\\\\Users", "/home/"):
            self.assertFalse(key in payload, f"Unexpected configuration content: {key}")
        self.assertIsNone(parsed["final_numeric_threshold"])
        self.assertFalse(parsed["architecture"]["identity_consistency_gate"])
        self.assertFalse(parsed["v2_final_evaluation_performed"])
        self.assertEqual(parsed["architecture"]["svm_parameters"]["C"], 4.)
        self.assertEqual(parsed["final_threshold_calibration_policy"]["diagnostic_far_targets"], [.2, .1, .05])

    def test_frozen_objects_and_deterministic_serialization(self):
        c = self.freeze()
        with self.assertRaises(FrozenInstanceError):
            c.architecture.selected_features = 512
        self.assertEqual(frozen_v2_configuration_to_json(c), frozen_v2_configuration_to_json(self.freeze()))
        for invalid in ("", "A" * 64, "/some/path"):
            with self.assertRaises(ConfigurationFreezeError):
                V2ReportProvenance(invalid, "a" * 64, "b" * 64)
        with self.assertRaises(ConfigurationFreezeError):
            replace(c, known_cross_session_folds=())


if __name__ == "__main__":
    unittest.main()
