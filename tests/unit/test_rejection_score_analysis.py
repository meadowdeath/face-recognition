from fractions import Fraction
import unittest

import numpy as np
from sklearn.metrics import roc_auc_score

from face_recognition.application.rejection_metrics import calibrate_threshold, evaluate_threshold, threshold_candidates
from face_recognition.application.rejection_score_analysis import analyze_scores, roc_auc_known_positive, score_distribution
from face_recognition.application.unknown_rejection_split import UnknownRejectionError


class RejectionScoreAnalysisTests(unittest.TestCase):
    def test_auc_continuous_known_positive_orientation_and_ties(self):
        self.assertEqual(roc_auc_known_positive(np.array([-1., -2.]), np.array([-5., -4.])), 1)
        self.assertEqual(roc_auc_known_positive(np.array([-5., -4.]), np.array([-1., -2.])), 0)
        self.assertEqual(roc_auc_known_positive(np.array([1., 1.]), np.array([1., 1.])), .5)
        known, unknown = np.array([.1, .4, .4, .9]), np.array([.2, .4, .5])
        expected = roc_auc_score(np.r_[np.ones(len(known)), np.zeros(len(unknown))], np.r_[known, unknown])
        self.assertAlmostEqual(roc_auc_known_positive(known, unknown), expected)

    def test_eer_exact_closest_point_ties_and_boundary_cases(self):
        known, unknown = np.array([-3., -1., 0., 1.]), np.array([-2., -1., .5])
        correct = np.array([True, False, True, True])
        analysis = analyze_scores(known, correct, unknown)
        thresholds = threshold_candidates(known, unknown, allow_negative=True)

        def key(threshold):
            far = Fraction(int((unknown >= threshold).sum()), len(unknown))
            frr = Fraction(int((known < threshold).sum()), len(known))
            return (abs(far - frr), far, frr, -threshold)

        expected = min(thresholds, key=key)
        point = analysis.equal_error_point
        self.assertEqual(point.threshold, expected)
        self.assertEqual(point.eer, (point.far + point.frr) / 2)
        self.assertEqual(analysis, analyze_scores(known, correct, unknown))
        same = analyze_scores(np.array([0., 0.]), np.array([True, True]), np.array([0., 0.]))
        self.assertEqual(same.equal_error_point.eer, .5)
        self.assertEqual(same.equal_error_point.far, 0)
        separated = analyze_scores(np.array([1., 2.]), np.array([True, True]), np.array([-2., -1.]))
        self.assertEqual(separated.equal_error_point.eer, 0)

    def test_all_far_targets_choose_highest_known_acceptance_then_lower_far(self):
        known = np.linspace(.5, 1.5, 20)
        unknown = np.linspace(0, 1, 20)
        correct = np.ones(20, dtype=bool)
        analysis = analyze_scores(known, correct, unknown)
        self.assertEqual(tuple(row.target_far for row in analysis.far_operating_points), (.2, .1, .05))
        thresholds = threshold_candidates(known, unknown, allow_negative=True)
        for point in analysis.far_operating_points:
            self.assertLessEqual(point.metrics.unknown_false_acceptance_rate, point.target_far)
            feasible = [t for t in thresholds if int((unknown >= t).sum()) * 100 <= round(point.target_far * 100) * len(unknown)]
            expected = max(feasible, key=lambda t: (int((known >= t).sum()), -int((unknown >= t).sum()), t))
            self.assertEqual(point.threshold, expected)
            self.assertEqual(point.status, "available")

    def test_impossible_useful_acceptance_reported_as_impractical_with_metrics(self):
        same = analyze_scores(np.zeros(10), np.ones(10, dtype=bool), np.zeros(20))
        for point in same.far_operating_points:
            self.assertEqual(point.status, "impractical")
            self.assertEqual(point.metrics.known_rejection_rate, 1)
            self.assertEqual(point.metrics.unknown_false_acceptance_rate, 0)
            self.assertIsNotNone(point.threshold)
        # Only one out of 100 known observations can be accepted at zero FAR.
        result = analyze_scores(np.r_[np.zeros(99), 2.], np.ones(100, dtype=bool), np.ones(20))
        self.assertTrue(all(point.status == "impractical" for point in result.far_operating_points))
        self.assertEqual(result.far_operating_points[0].metrics.known_acceptance_rate, .01)

    def test_distribution_population_std_and_linear_percentiles(self):
        actual = score_distribution(np.array([-4., -2., 0., 2., 4.]))
        self.assertEqual(actual.count, 5)
        self.assertEqual(actual.mean, 0)
        self.assertAlmostEqual(actual.std, np.sqrt(8))
        self.assertEqual(actual.median, 0)
        self.assertAlmostEqual(actual.p10, -3.2)
        self.assertEqual(actual.p25, -2)
        self.assertEqual(actual.p75, 2)
        self.assertAlmostEqual(actual.p90, 3.2)

    def test_balanced_threshold_metrics_reuse_previous_calculation(self):
        known, unknown = np.array([.9, .1, .8]), np.array([.2, .8])
        correct = np.array([True, True, False])
        analysis = analyze_scores(known, correct, unknown)
        self.assertEqual(analysis.calibration, calibrate_threshold(known, correct, unknown))
        self.assertEqual(analysis.calibration.metrics,
                         evaluate_threshold(known, correct, unknown, analysis.calibration.threshold))

    def test_invalid_empty_nonfinite_analysis_inputs_fail(self):
        for values in (np.array([]), np.array([np.nan]), np.array([np.inf])):
            with self.assertRaises(UnknownRejectionError):
                score_distribution(values)
            with self.assertRaises(UnknownRejectionError):
                roc_auc_known_positive(values, np.array([0.]))
