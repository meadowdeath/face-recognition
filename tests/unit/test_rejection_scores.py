import unittest

import numpy as np

from face_recognition.application.rejection_metrics import calibrate_threshold, decision_margin, evaluate_threshold
from face_recognition.application.unknown_rejection_split import UnknownRejectionError
from face_recognition.domain.models.rejection_score_evaluation import CENTROID_SCALE_EPSILON
from face_recognition.infrastructure.features.rejection_scores import PredictedClassCentroidScorer, decision_top1


class RejectionScoreTests(unittest.TestCase):
    def test_svm_top1_and_margin_in_fixed_class_order(self):
        scores = np.array([[1., 4., 2.], [3., 3., 1.], [-5., -2., -4.]])
        self.assertTrue(np.array_equal(decision_top1(scores), np.array([4., 3., -2.])))
        self.assertTrue(np.array_equal(decision_margin(scores), np.array([2., 0., 2.])))
        self.assertTrue(np.array_equal(decision_top1(scores[:, ::-1]), decision_top1(scores)))

    def test_top1_rejects_nonfinite_and_invalid_score_columns(self):
        for values in (np.ones((2, 1)), np.ones(2), np.array([[1, np.nan]]), np.array([[np.inf, 1]])):
            with self.assertRaises(UnknownRejectionError):
                decision_top1(values)

    def test_centroids_and_class_median_scales_use_true_fit_labels(self):
        X = np.array([[0., 0.], [2., 0.], [10., 0.], [14., 0.]], dtype=np.float32)
        y = np.array([0, 0, 1, 1], dtype=np.int32)
        scorer = PredictedClassCentroidScorer().fit(X, y, 2)
        query = np.array([[2., 0.], [2., 0.], [14., 0.]])
        predicted = np.array([0, 1, 1])
        raw, normalized = scorer.score_pair(query, predicted)
        self.assertTrue(np.allclose(raw, np.array([-1., -10., -2.])))
        self.assertTrue(np.allclose(normalized, np.array([-1., -5., -1.])))
        # The second query uses its predicted class center, NOT the nearest center.
        self.assertEqual(scorer.fit_samples, 4)
        self.assertNotIn("array", repr(scorer))

    def test_even_median_scale_and_zero_scale_epsilon_are_deterministic(self):
        scorer = PredictedClassCentroidScorer().fit(np.array([[0.], [0.], [10.], [10.]]), np.array([0, 0, 1, 1]), 2)
        query, predicted = np.array([[1.], [10.]]), np.array([0, 1])
        actual = scorer.score(query, predicted, normalized=True)
        self.assertTrue(np.allclose(actual, np.array([-1 / CENTROID_SCALE_EPSILON, 0.])))
        self.assertTrue(np.array_equal(actual, scorer.score(query, predicted, normalized=True)))
        varied = PredictedClassCentroidScorer().fit(np.array([[0.], [1.], [2.], [9.], [10.], [20.]]),
                                                   np.array([0, 0, 0, 0, 1, 1]), 2)
        # Class 0 center is 3, distances [3,2,1,6], median 2.5.
        self.assertTrue(np.allclose(varied.score(np.array([[8.]]), np.array([0]), normalized=True), np.array([-2.])))

    def test_scoring_does_not_modify_fit_stats_or_input_and_never_refits(self):
        X, y = np.array([[0.], [2.], [10.], [14.]], dtype=np.float32), np.array([0, 0, 1, 1])
        before = X.tobytes()
        scorer = PredictedClassCentroidScorer().fit(X, y, 2)
        reference = scorer.score(X, y)
        scorer.score(np.array([[1000.]]), np.array([0]))
        self.assertTrue(np.array_equal(reference, scorer.score(X, y)))
        self.assertTrue(before == X.tobytes())

    def test_invalid_centroid_data_predicted_labels_and_unfitted_usage_fail(self):
        with self.assertRaises(UnknownRejectionError):
            PredictedClassCentroidScorer().score(np.ones((1, 1)), np.array([0]))
        for X, y in ((np.array([[np.nan], [1.]]), np.array([0, 1])),
                     (np.ones((2, 1)), np.array([0, 0])), (np.ones((2, 1)), np.array([0., 1.]))):
            with self.assertRaises(UnknownRejectionError):
                PredictedClassCentroidScorer().fit(X, y, 2)
        scorer = PredictedClassCentroidScorer().fit(np.array([[0.], [1.]]), np.array([0, 1]), 2)
        for X, predicted in ((np.array([[np.inf]]), np.array([0])), (np.ones((1, 1)), np.array([2])),
                             (np.ones((1, 2)), np.array([0]))):
            with self.assertRaises(UnknownRejectionError):
                scorer.score(X, predicted)

    def test_signed_distance_thresholds_and_original_margin_defaults(self):
        known, unknown = np.array([-1., -2.]), np.array([-5., -4.])
        correct = np.array([True, False])
        result = calibrate_threshold(known, correct, unknown, allow_negative=True)
        self.assertEqual(result.threshold, -2.)
        self.assertEqual(result.metrics.binary_balanced_accuracy, 1)
        self.assertEqual(result.metrics.known_correct_identification_rate, .5)
        self.assertEqual(result.metrics.known_misidentification_rate, .5)
        with self.assertRaises(UnknownRejectionError):
            calibrate_threshold(known, correct, unknown)
        positive_known, positive_unknown = np.array([.9, .8]), np.array([.1, .2])
        self.assertEqual(calibrate_threshold(positive_known, correct, positive_unknown),
                         calibrate_threshold(positive_known, correct, positive_unknown, allow_negative=True))
        self.assertEqual(result.metrics, evaluate_threshold(known, correct, unknown, -2., allow_negative=True))
