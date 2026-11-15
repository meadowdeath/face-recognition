import unittest
import numpy as np
from sklearn.covariance import LedoitWolf
from face_recognition.domain.models.rejection import RejectionError
from face_recognition.infrastructure.features.mahalanobis import PredictedClassMahalanobisScorer


class MahalanobisTests(unittest.TestCase):
    def test_regularized_singular_high_dimensional_and_constant_classes(self):
        X = np.zeros((6, 20), dtype=np.float32)
        X[3:] = 10.
        y = np.array([0, 0, 0, 1, 1, 1])
        scorer = PredictedClassMahalanobisScorer().fit(X, y, 2)
        values = scorer.score(np.array([np.zeros(20), np.ones(20)]), np.array([0, 0]))
        self.assertEqual(values[0], 0.)
        self.assertTrue(np.isfinite(values).all())
        self.assertLess(values[1], 0.)
        np.testing.assert_array_equal(values, scorer.score(np.array([np.zeros(20), np.ones(20)]), np.array([0, 0])))
        for bad in (np.nan, np.inf):
            with self.assertRaises(RejectionError):
                scorer.score(np.full((1, 20), bad), np.array([0]))
        self.assertNotIn("array", repr(scorer))

    def test_negative_distance_uses_predicted_class_distribution_and_frozen_ridge(self):
        X = np.array([[0., 0.], [2., 0.], [10., 0.], [14., 0.]])
        y = np.array([0, 0, 1, 1])
        scorer = PredictedClassMahalanobisScorer().fit(X, y, 2)
        actual = scorer.score(np.array([[2., 0.]]), np.array([1]))[0]
        estimate = LedoitWolf(store_precision=False).fit(X[y == 1])
        covariance = estimate.covariance_ + np.eye(2) * 1e-8 * max(1., np.trace(estimate.covariance_) / 2)
        delta = np.array([2., 0.]) - estimate.location_
        self.assertAlmostEqual(actual, -np.sqrt(delta @ np.linalg.solve(covariance, delta)))
        scores = scorer.score(np.array([[1., 0.], [100., 0.]]), np.array([0, 0]))
        self.assertGreater(scores[0], scores[1])
        self.assertTrue(np.all(scores <= 0))
