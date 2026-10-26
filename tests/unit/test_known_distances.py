import unittest
from unittest.mock import patch

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

from face_recognition.application.unknown_rejection_split import UnknownRejectionError
from face_recognition.infrastructure.features.known_distances import KnownDistanceScorer


class KnownDistanceTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[0., 0.], [3., 4.], [6., 8.], [9., 12.], [12., 16.], [15., 20.]], dtype=np.float32)
        self.y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)

    def test_exact_euclidean_nearest_three_and_five_scores(self):
        result = KnownDistanceScorer().fit(self.X, self.y, 2).score(self.X[:2])
        self.assertEqual(tuple(name for name, _ in result.confidence), ("nearest_1", "mean_3nn", "mean_5nn"))
        expected = (np.array([0., 0.]), np.array([-5., -10 / 3]), np.array([-10., -7.]))
        for (_, actual), target in zip(result.confidence, expected):
            self.assertTrue(np.allclose(actual, target))
            self.assertTrue(np.all(actual <= 0))
        self.assertTrue(np.array_equal(result.nearest_labels, np.array([0, 0])))

    def test_score_direction_larger_means_closer_to_known(self):
        result = KnownDistanceScorer().fit(self.X, self.y, 2).score(np.array([[0., 0.], [100., 100.]]))
        self.assertTrue(all(values[0] > values[1] for _, values in result.confidence))

    def test_batched_results_match_exhaustive_distances_and_deterministic_repeats(self):
        query = np.arange(28, dtype=np.float32).reshape(14, 2)
        scorer = KnownDistanceScorer(batch_size=2).fit(self.X, self.y, 2)
        calls = []

        def distance(X, references):
            calls.append((X.shape, references.shape))
            return euclidean_distances(X, references)

        with patch("face_recognition.infrastructure.features.known_distances.euclidean_distances", distance):
            output = scorer.score(query)
        self.assertEqual(len(calls), 7)
        self.assertTrue(all(shape[0][0] <= 2 and shape[1] == (6, 2) for shape in calls))
        actual_distances = np.sort(np.linalg.norm(query[:, None].astype(np.float64) - self.X[None, :], axis=2), axis=1)
        for (_, actual), count in zip(output.confidence, (1, 3, 5)):
            self.assertTrue(np.allclose(actual, -actual_distances[:, :count].mean(axis=1)))
        other = KnownDistanceScorer().fit(self.X, self.y, 2).score(query)
        self.assertTrue(all(np.allclose(a, b) for (_, a), (_, b) in zip(output.confidence, other.confidence)))
        self.assertTrue(all(np.array_equal(a, b) for (_, a), (_, b) in zip(output.confidence, scorer.score(query).confidence)))

    def test_exact_nearest_ties_choose_first_reference_row(self):
        X, y = np.arange(0., 12., 2).reshape(-1, 1), np.array([0, 1, 0, 1, 0, 1])
        result = KnownDistanceScorer().fit(X, y, 2).score(np.array([[1.], [3.]]))
        self.assertTrue(np.array_equal(result.nearest_labels, np.array([0, 1])))

    def test_insufficient_reference_samples_rejected_for_joint_experiment(self):
        for count in (1, 2, 3, 4):
            with self.assertRaisesRegex(UnknownRejectionError, "At least 5 known fit reference"):
                KnownDistanceScorer().fit(self.X[:count], self.y[:count], 2)

    def test_invalid_fit_query_or_batch_size_rejected_without_values(self):
        for size in (0, 129, True, 1.5):
            with self.assertRaises(UnknownRejectionError):
                KnownDistanceScorer(batch_size=size)
        for invalid in (np.nan, np.inf, -np.inf):
            X = self.X.copy()
            X[0, 0] = invalid
            with self.assertRaises(UnknownRejectionError):
                KnownDistanceScorer().fit(X, self.y, 2)
            with self.assertRaises(UnknownRejectionError):
                KnownDistanceScorer().fit(self.X, self.y, 2).score(X)
        with self.assertRaises(UnknownRejectionError):
            KnownDistanceScorer().score(self.X)
        scorer = KnownDistanceScorer().fit(self.X, self.y, 2)
        for query in (np.ones((2, 3)), np.empty((0, 2)), np.array([["PRIVATE", "VALUE"]])):
            with self.assertRaises(UnknownRejectionError) as caught:
                scorer.score(query)
            self.assertNotIn("PRIVATE", str(caught.exception))
        with self.assertRaises(UnknownRejectionError):
            KnownDistanceScorer().fit(self.X, np.zeros(6, dtype=int), 2)

    def test_reference_owned_readonly_and_input_not_mutated(self):
        before, labels = self.X.tobytes(), self.y.tobytes()
        scorer = KnownDistanceScorer().fit(self.X, self.y, 2)
        query = np.array([[1., 2.]])
        initial = scorer.score(query)
        self.assertFalse(scorer._reference.flags.writeable)
        self.assertFalse(scorer._labels.flags.writeable)
        self.assertFalse(np.shares_memory(self.X, scorer._reference))
        self.assertEqual(self.X.tobytes(), before)
        self.assertEqual(self.y.tobytes(), labels)
        self.X.fill(100)
        self.y.fill(1)
        self.assertTrue(all(np.array_equal(a, b) for (_, a), (_, b) in zip(initial.confidence, scorer.score(query).confidence)))
        self.assertNotIn("array", repr(scorer))
        self.assertNotIn("array", repr(initial))

    def test_requested_one_five_subset_preserves_original_scores_without_mean_three(self):
        query = self.X[:2]
        original = dict(KnownDistanceScorer().fit(self.X, self.y, 2).score(query).confidence)
        scorer = KnownDistanceScorer(neighbor_counts=(5, 1)).fit(self.X, self.y, 2)
        result = scorer.score(query)
        self.assertEqual(scorer._scores, (("nearest_1", 1), ("mean_5nn", 5)))
        self.assertEqual(tuple(name for name, _ in result.confidence), ("nearest_1", "mean_5nn"))
        self.assertTrue(all(np.array_equal(values, original[name]) for name, values in result.confidence))

    def test_only_nonempty_unique_subsets_of_existing_neighbor_counts_are_allowed(self):
        for counts in ((), (2,), (7,), (1, 1), (True,), (1, 3, 5, 9)):
            with self.assertRaises(UnknownRejectionError):
                KnownDistanceScorer(neighbor_counts=counts)
        scorer = KnownDistanceScorer(neighbor_counts=(1,)).fit(self.X[:2], np.array([0, 1]), 2)
        self.assertEqual(tuple(name for name, _ in scorer.score(self.X[:1]).confidence), ("nearest_1",))
