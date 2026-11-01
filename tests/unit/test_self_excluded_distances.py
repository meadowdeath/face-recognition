import unittest

import numpy as np

from face_recognition.application.unknown_rejection_split import UnknownRejectionError
from face_recognition.infrastructure.features.known_distances import KnownDistanceScorer


class SelfExcludedDistancesTests(unittest.TestCase):
    def setUp(self):
        self.X = np.arange(6., dtype=np.float32).reshape(-1, 1)
        self.y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
        self.scorer = KnownDistanceScorer(neighbor_counts=(5,), batch_size=2).fit(self.X, self.y, 2)

    def test_zero_self_distance_excluded_and_exactly_five_others_used(self):
        ordinary = self.scorer.score(self.X[:1]).confidence[0][1]
        excluded = self.scorer.score(self.X[:1], exclude_reference_indices=np.array([0])).confidence[0][1]
        self.assertEqual(ordinary.item(), -2.)
        self.assertEqual(excluded.item(), -3.)

    def test_each_row_matches_explicit_leave_one_out_distances(self):
        actual = self.scorer.score(self.X, exclude_reference_indices=np.arange(6)).confidence[0][1]
        distances = np.abs(self.X.astype(np.float64) - self.X.T)
        np.fill_diagonal(distances, np.inf)
        expected = -np.sort(distances, axis=1)[:, :5].mean(axis=1)
        self.assertTrue(np.allclose(actual, expected))
        order = np.array([4, 1, 5, 0, 3, 2])
        shuffled = self.scorer.score(self.X[order], exclude_reference_indices=order).confidence[0][1]
        self.assertTrue(np.array_equal(shuffled, actual[order]))

    def test_duplicate_vectors_remain_eligible_except_exact_source_row(self):
        X = np.array([[0.], [0.], [1.], [2.], [3.], [4.]])
        scorer = KnownDistanceScorer(neighbor_counts=(5,)).fit(X, self.y, 2)
        result = scorer.score(X[:2], exclude_reference_indices=np.array([0, 1])).confidence[0][1]
        self.assertTrue(np.array_equal(result, np.array([-2., -2.])))
        self.assertTrue(np.array_equal(result, scorer.score(X[:2], exclude_reference_indices=np.array([0, 1])).confidence[0][1]))

    def test_at_least_five_other_references_required(self):
        scorer = KnownDistanceScorer(neighbor_counts=(5,)).fit(self.X[:5], self.y[:5], 2)
        with self.assertRaisesRegex(UnknownRejectionError, "OTHER references"):
            scorer.score(self.X[:1], exclude_reference_indices=np.array([0]))

    def test_invalid_indices_or_mismatched_query_rejected(self):
        for indices in (np.array([-1]), np.array([6]), np.array([0.]), np.array([True]), np.array([0, 1])):
            with self.assertRaises(UnknownRejectionError):
                self.scorer.score(self.X[:1], exclude_reference_indices=indices)
        with self.assertRaisesRegex(UnknownRejectionError, "must match"):
            self.scorer.score(self.X[:1], exclude_reference_indices=np.array([1]))

    def test_scoring_does_not_mutate_reference_or_query(self):
        reference, labels = self.scorer.reference_arrays()
        before, query = reference.tobytes(), self.X.tobytes()
        self.scorer.score(self.X, exclude_reference_indices=np.arange(6))
        self.assertTrue(before == reference.tobytes() and query == self.X.tobytes())
        self.assertFalse(reference.flags.writeable or labels.flags.writeable)
