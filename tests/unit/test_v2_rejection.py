from dataclasses import asdict
import unittest

import numpy as np

from face_recognition.application.rejection_metrics import calibrate_threshold
from face_recognition.application.v2_rejection import (
    binary_matrix, calibrate, configurations, confusion, distribution, far_points, identity_thresholds,
    matrix_metrics, open_matrix, rank_key, rotation_masks,
)
from face_recognition.application.v2_study_data import V2StudyError


class V2RejectionTests(unittest.TestCase):
    def test_exact_strategy_and_window_grid(self):
        rows = configurations()
        self.assertEqual(len(rows), 42)
        self.assertEqual(len({r.identifier for r in rows}), 42)
        self.assertEqual({r.window for r in rows}, {1, 3, 5, 7})
        self.assertEqual(sum(r.per_identity for r in rows), 14)
        self.assertFalse(any(r.strict_majority_gate for r in rows if r.window == 1))

    def test_no_gate_exactly_preserves_existing_threshold_rule(self):
        k, u, c = np.array([.2, .6, .7, .7]), np.array([.1, .3, .8]), np.array([True, False, True, True])
        previous = calibrate_threshold(k, c, u)
        result = calibrate(k, c, u, np.ones(4, bool), np.ones(3, bool))
        self.assertEqual(result[0], previous.threshold)
        self.assertEqual(result[3], previous.metrics)

    def test_gate_rejections_are_in_denominators_and_threshold_tie_break(self):
        k, u = np.array([.9, .8]), np.array([.7, .1])
        result = calibrate(k, np.ones(2, bool), u, np.array([True, False]), np.array([False, False]))
        self.assertEqual(result[3].known_rejection_rate, .5)
        self.assertEqual(result[3].unknown_rejection_rate, 1)
        self.assertEqual(result[0], .9)  # Highest tied threshold accepting the one eligible known.

    def test_gate_all_ineligible_is_explicit_reject_all(self):
        result = calibrate(np.array([1.]), np.array([True]), np.array([2.]), np.array([False]), np.array([False]))
        self.assertEqual(result[3].known_rejection_rate, 1)
        self.assertEqual(result[3].binary_balanced_accuracy, .5)

    def test_per_identity_support_fallback_and_predicted_identity_lookup(self):
        k, u = np.r_[np.full(10, 3.), 5.], np.r_[np.ones(10), 4.]
        kp, up = np.r_[np.zeros(10, int), 1], np.r_[np.zeros(10, int), 1]
        rows = identity_thresholds(("a", "b"), k, np.ones(11, bool), u, kp, up,
                                   np.ones(11, bool), np.ones(11, bool), 2.)
        self.assertTrue(rows[0].available)
        self.assertEqual(rows[0].used_threshold, 3.)
        self.assertFalse(rows[1].available)
        self.assertIsNone(rows[1].threshold)
        self.assertEqual(rows[1].used_threshold, 2.)
        gated = np.ones(11, bool)
        gated[0] = False
        rows = identity_thresholds(("a", "b"), k, np.ones(11, bool), u, kp, up, gated, gated, 2.)
        self.assertFalse(rows[0].available)

    def test_unknown_rotations_leave_whole_identity_out(self):
        people = np.array(["z", "a", "z", "b", "a"])
        rotations = rotation_masks(people)
        coverage = np.zeros(len(people), int)
        for cal, test in rotations:
            self.assertFalse(bool(np.any(cal & test)))
            self.assertFalse(set(people[cal]) & set(people[test]))
            coverage += test
        self.assertTrue(bool(np.all(coverage == 1)))
        self.assertEqual(people[rotations[0][1]][0], "a")

    def test_open_binary_confusion_and_exact_metrics(self):
        m = open_matrix(np.array([0, 0, 1]), np.array([0, 1, 1]), np.array([True, True, False]),
                        np.array([0, 1]), np.array([False, True]), 2)
        self.assertEqual(m.tolist(), [[1, 1, 0], [0, 0, 1], [0, 1, 1]])
        self.assertEqual(binary_matrix(m).tolist(), [[2, 1], [1, 1]])
        metrics = matrix_metrics(m)
        self.assertEqual(metrics.known_correct_identification_rate, 1 / 3)
        self.assertEqual(metrics.known_misidentification_rate, 1 / 3)
        self.assertEqual(metrics.known_rejection_rate, 1 / 3)
        self.assertEqual(metrics.unknown_false_acceptance_rate, .5)
        self.assertEqual(metrics.open_set_exact_accuracy, .4)
        self.assertAlmostEqual(metrics.binary_balanced_accuracy, (2 / 3 + .5) / 2)
        summary = confusion(m, ("a", "b", "UNKNOWN"), 2)
        self.assertEqual(summary.role_adjusted_counts[0], (.5, .5, 0.))
        self.assertEqual(summary.row_normalized_rates[0], (.5, .5, 0.))

    def test_far_points_and_impractical(self):
        points = far_points(np.array([1., 2.]), np.ones(2, bool), np.array([3., 4.]), np.ones(2, bool), np.ones(2, bool))
        self.assertEqual([p["far_target"] for p in points], [.2, .1, .05])
        self.assertTrue(all(p["status"] == "impractical" for p in points))
        self.assertTrue(all(p["calibration_metrics"]["unknown_false_acceptance_rate"] == 0 for p in points))

    def test_distribution_population_std_percentiles_and_empty(self):
        d = distribution(np.array([0., 1., 2., 3.]))
        self.assertEqual(d.count, 4)
        stats = asdict(d.statistics)
        self.assertIn(1.5, stats.values())
        self.assertIn(float(np.std([0, 1, 2, 3])), stats.values())
        self.assertEqual(distribution(np.array([])).statistics, None)

    def test_nonfinite_rejected_without_values(self):
        with self.assertRaises(ValueError):
            calibrate(np.array([np.nan]), np.array([True]), np.array([2.]), np.ones(1, bool), np.ones(1, bool))
        with self.assertRaises(V2StudyError):
            calibrate(np.array([1.]), np.array([1]), np.array([2.]), np.ones(1, bool), np.ones(1, bool))

    def test_ranking_metric_priority_then_simplicity_then_window(self):
        base = {"mean_roc_auc": .8, "mean_eer": .2, "metrics": asdict(matrix_metrics(np.array([[9, 1], [1, 9]]))),
                "configuration": asdict(configurations()[0])}
        import copy
        for location, key, value in ((None, "mean_roc_auc", .81), (None, "mean_eer", .19),
                                     ("metrics", "unknown_false_acceptance_rate", .01),
                                     ("metrics", "known_rejection_rate", .01),
                                     ("metrics", "known_correct_identification_rate", .99),
                                     ("metrics", "open_set_exact_accuracy", .99),
                                     ("metrics", "known_misidentification_rate", -.01)):
            other = copy.deepcopy(base)
            (other if location is None else other[location])[key] = value
            self.assertLess(rank_key(other), rank_key(base))
        for key, worse in (("per_identity", True), ("strict_majority_gate", True), ("window", 7)):
            other = copy.deepcopy(base)
            other["configuration"][key] = worse
            self.assertLess(rank_key(base), rank_key(other))


if __name__ == "__main__":
    unittest.main()
