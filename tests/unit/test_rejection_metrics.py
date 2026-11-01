from fractions import Fraction
import math
import unittest

import numpy as np

from face_recognition.application.rejection_metrics import calibrate_threshold, decision_margin, evaluate_threshold, threshold_candidates
from face_recognition.application.unknown_rejection_split import UnknownRejectionError


class RejectionMetricsTests(unittest.TestCase):
    def test_top_two_margin_and_class_column_order_invariance(self):
        scores = np.array([[1, 4, 2], [3, 3, 1], [-5, -2, -4]], dtype=float)
        self.assertTrue(np.array_equal(decision_margin(scores), np.array([2, 0, 2])))
        self.assertTrue(np.array_equal(decision_margin(scores[:, [2, 0, 1]]), np.array([2, 0, 2])))
        self.assertTrue(np.array_equal(decision_margin(np.array([[-2, 2], [1, -1]])), np.array([4, 2])))

    def test_invalid_or_nonfinite_scores_rejected(self):
        for scores in (np.ones((2, 1)), np.ones(2), np.ones((0, 3)), np.array([[0, np.nan]]), np.array([[0, np.inf]])):
            with self.assertRaises(UnknownRejectionError):
                decision_margin(scores)

    def test_exact_open_set_cases_rates_and_binary_metrics(self):
        # Known: accepted correct, rejected correct, accepted wrong.
        # Unknown: rejected, falsely accepted. Equality at the threshold accepts.
        metrics = evaluate_threshold(np.array([.9, .1, .8]), np.array([True, True, False]), np.array([.2, .8]), .8)
        self.assertAlmostEqual(metrics.known_acceptance_rate, 2 / 3)
        self.assertAlmostEqual(metrics.known_rejection_rate, 1 / 3)
        self.assertEqual(metrics.unknown_rejection_rate, .5)
        self.assertEqual(metrics.unknown_false_acceptance_rate, .5)
        self.assertAlmostEqual(metrics.binary_balanced_accuracy, (2 / 3 + .5) / 2)
        self.assertAlmostEqual(metrics.binary_macro_f1, ((4 / 6) + (2 / 4)) / 2)
        self.assertAlmostEqual(metrics.known_correct_identification_rate, 1 / 3)
        self.assertAlmostEqual(metrics.known_misidentification_rate, 1 / 3)
        self.assertEqual(metrics.open_set_exact_accuracy, 2 / 5)

    def test_threshold_candidates_include_values_midpoints_and_finite_boundaries(self):
        thresholds = threshold_candidates(np.array([1., 3.]), np.array([2., 3.]))
        self.assertTrue(np.array_equal(thresholds[:-1], np.array([0., 1., 1.5, 2., 2.5, 3.])))
        self.assertGreater(thresholds[-1], 3)
        self.assertTrue(np.isfinite(thresholds).all())
        self.assertTrue(np.array_equal(thresholds, threshold_candidates(np.array([3., 1.]), np.array([3., 2.]))))

    def test_perfect_separation_and_higher_threshold_tie_rule(self):
        known, unknown = np.array([.8, .9]), np.array([.1, .2])
        result = calibrate_threshold(known, np.array([True, True]), unknown)
        self.assertEqual(result.threshold, .8)  # Midpoint and .8 share outcomes; larger wins.
        self.assertEqual(result.metrics.binary_balanced_accuracy, 1)
        self.assertEqual(result.metrics.open_set_exact_accuracy, 1)
        self.assertEqual(result.metrics.unknown_false_acceptance_rate, 0)
        self.assertEqual(result.metrics.known_rejection_rate, 0)

    def test_indistinguishable_confidences_prefer_lower_far_then_reject_all(self):
        result = calibrate_threshold(np.array([0., 0.]), np.array([True, False]), np.array([0., 0.]))
        self.assertEqual(result.threshold, math.nextafter(0., math.inf))
        self.assertEqual(result.metrics.binary_balanced_accuracy, .5)
        self.assertEqual(result.metrics.unknown_false_acceptance_rate, 0)
        self.assertEqual(result.metrics.known_rejection_rate, 1)

    def test_search_matches_brute_force_exact_rational_tie_breaking(self):
        known, unknown = np.array([0., .2, .4, .4, .9]), np.array([0., .3, .4])
        correct = np.array([True, True, False, True, False])
        candidates = threshold_candidates(known, unknown)

        def key(threshold):
            ka, ur = int((known >= threshold).sum()), int((unknown < threshold).sum())
            return ((Fraction(ka, len(known)) + Fraction(ur, len(unknown))) / 2,
                    -Fraction(len(unknown) - ur, len(unknown)), -Fraction(len(known) - ka, len(known)), threshold)

        expected = max(candidates, key=key)
        actual = calibrate_threshold(known, correct, unknown)
        self.assertEqual(actual.threshold, expected)
        self.assertEqual(actual.metrics, evaluate_threshold(known, correct, unknown, expected))

    def test_empty_nonfinite_negative_and_mismatched_observations_fail(self):
        for known, correct, unknown in (([], [], [.1]), ([.2], [True], []), ([np.nan], [True], [.1]),
                                         ([-.1], [True], [.1]), ([.2], [True, False], [.1]), ([.2], [1], [.1])):
            with self.assertRaises(UnknownRejectionError):
                calibrate_threshold(np.array(known), np.array(correct), np.array(unknown))
