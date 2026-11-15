from fractions import Fraction
import unittest

import numpy as np
from sklearn.metrics import roc_auc_score

from face_recognition.application.rejection_metrics import threshold_candidates
from face_recognition.application.rejection_score_analysis import score_separation, roc_auc_known_positive, score_distribution
from face_recognition.domain.models.rejection import RejectionError


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
        analysis = score_separation(known, unknown)
        thresholds = threshold_candidates(known, unknown, allow_negative=True)

        def key(threshold):
            far = Fraction(int((unknown >= threshold).sum()), len(unknown))
            frr = Fraction(int((known < threshold).sum()), len(known))
            return (abs(far - frr), far, frr, -threshold)

        expected = min(thresholds, key=key)
        point = analysis.equal_error_point
        self.assertEqual(point.threshold, expected)
        self.assertEqual(point.eer, (point.far + point.frr) / 2)
        self.assertEqual(analysis, score_separation(known, unknown))
        same = score_separation(np.array([0., 0.]), np.array([0., 0.]))
        self.assertEqual(same.equal_error_point.eer, .5)
        self.assertEqual(same.equal_error_point.far, 0)
        separated = score_separation(np.array([1., 2.]), np.array([-2., -1.]))
        self.assertEqual(separated.equal_error_point.eer, 0)

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

    def test_invalid_empty_nonfinite_analysis_inputs_fail(self):
        for values in (np.array([]), np.array([np.nan]), np.array([np.inf])):
            with self.assertRaises(RejectionError):
                score_distribution(values)
            with self.assertRaises(RejectionError):
                roc_auc_known_positive(values, np.array([0.]))
