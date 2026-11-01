from dataclasses import replace
import unittest

import numpy as np

from face_recognition.application.temporal_windows import aggregate_windows, build_windows, vote_identity
from face_recognition.application.unknown_rejection_split import UnknownRejectionError
from face_recognition.domain.models.identity_split import IdentityPartition, SplitSampleReference


def references(count, *, person="synthetic_a", session="visit_a", condition="center", start=0):
    return tuple(SplitSampleReference(person, session, condition, index, IdentityPartition.KNOWN_TRAIN_POOL,
                                      f"all_xyz_v1/{person}/{session}/{condition}.jsonl")
                 for index in range(start, start + count))


class TemporalWindowTests(unittest.TestCase):
    def test_exact_sizes_nonoverlap_and_trailing_counts(self):
        samples = references(20, start=4)
        for size in (1, 3, 5, 7):
            plan = build_windows(samples, size)
            indices = [index for window in plan.windows for index in window.indices]
            self.assertEqual(indices, list(range(20 // size * size)))
            self.assertEqual(len(set(indices)), len(indices))
            self.assertEqual(plan.counts.windows, 20 // size)
            self.assertEqual(plan.counts.discarded_samples, 20 % size)
            self.assertEqual(plan.counts.used_samples + plan.counts.discarded_samples, 20)
            self.assertTrue(all(len(window.indices) == size for window in plan.windows))

    def test_shuffled_source_order_is_sorted_by_sample_index(self):
        samples = tuple(reversed(references(8, start=9)))
        plan = build_windows(samples, 3)
        self.assertEqual(tuple(tuple(samples[i].sample_index for i in window.indices) for window in plan.windows),
                         ((9, 10, 11), (12, 13, 14)))
        self.assertEqual(plan, build_windows(samples, 3))

    def test_identity_session_and_condition_boundaries_never_cross(self):
        samples = tuple(sample for person in ("synthetic_b", "synthetic_a")
                        for session in ("visit_b", "visit_a") for condition in ("near", "center")
                        for sample in references(8, person=person, session=session, condition=condition))
        plan = build_windows(tuple(reversed(samples)), 7)
        reversed_samples = tuple(reversed(samples))
        self.assertEqual(plan.counts.windows, 8)
        self.assertEqual(plan.counts.discarded_samples, 8)
        for window in plan.windows:
            keys = {(reversed_samples[i].person_id, reversed_samples[i].session_id, reversed_samples[i].condition)
                    for i in window.indices}
            self.assertEqual(keys, {(window.person_id, window.session_id, window.condition)})
        self.assertEqual([name for name, _ in plan.condition_counts], ["center", "near"])
        self.assertEqual([c.windows for _, c in plan.condition_counts], [4, 4])

    def test_short_groups_not_padded_or_joined(self):
        plan = build_windows(references(2) + references(2, condition="near"), 3)
        self.assertEqual(plan.windows, ())
        self.assertEqual(plan.counts.discarded_samples, 4)
        self.assertEqual(build_windows((), 7).counts.source_samples, 0)

    def test_duplicates_gaps_and_unsupported_sizes_rejected(self):
        for samples in (references(3) + (references(1)[0],), (references(3)[0], references(3)[2])):
            with self.assertRaisesRegex(UnknownRejectionError, "unique contiguous"):
                build_windows(samples, 1)
        for size in (0, 2, 4, 9, True):
            with self.assertRaises(UnknownRejectionError):
                build_windows(references(8), size)
        for sample in (replace(references(1)[0], sample_index=-1), replace(references(1)[0], condition="invalid")):
            with self.assertRaises(UnknownRejectionError):
                build_windows((sample,), 1)

    def test_unanimous_and_unique_plurality_votes(self):
        for predictions, winner, category in (([1, 1, 1], 1, "unanimous"), ([2, 1, 2], 2, "majority"),
                                               ([0, 0, 0, 1, 1, 2, 2], 0, "majority")):
            self.assertEqual(vote_identity(np.array(predictions), np.zeros((len(predictions), 3))), (winner, category))

    def test_vote_count_precedes_decision_sum(self):
        self.assertEqual(vote_identity(np.array([0, 0, 1]), np.array([[0, 99], [0, 99], [0, 99]])), (0, "majority"))

    def test_tie_uses_all_frame_scores_then_class_order(self):
        # Three tied labels; class 1 wins on the score from another label's frame.
        self.assertEqual(vote_identity(np.array([0, 1, 2]), np.array([[5, 0, 0], [0, 4, 0], [0, 10, 3]])), (1, "tie"))
        self.assertEqual(vote_identity(np.array([2, 1, 0]), np.ones((3, 3))), (0, "tie"))

    def test_median_uses_all_scores_and_preserves_signed_direction(self):
        plan = build_windows(references(3), 3)
        result = aggregate_windows(plan, np.array([0, 0, 1]), np.ones((3, 2)),
                                   (("centroid_distance", np.array([-9., -3., -1.])),
                                    ("svm_margin", np.array([1., 100., 2.]))))
        self.assertEqual(result.predicted.tolist(), [0])
        self.assertEqual([v.item() for _, v in result.confidence], [-3., 2.])

    def test_vote_stability_categories_are_exclusive(self):
        plan = build_windows(references(9), 3)
        predicted = np.array([0, 0, 0, 0, 0, 1, 0, 1, 2])
        result = aggregate_windows(plan, predicted, np.ones((9, 3)), (("svm_margin", np.ones(9)),))
        stability = result.vote_stability
        self.assertEqual((stability.unanimous_windows, stability.majority_not_unanimous_windows, stability.tied_windows), (1, 1, 1))
        self.assertEqual(tuple(vars(stability.rates).values()), (1 / 3, 1 / 3, 1 / 3))

    def test_nonfinite_scores_or_invalid_discarded_prediction_rejected(self):
        plan = build_windows(references(4), 3)
        for invalid in (np.nan, np.inf, -np.inf):
            values = np.array([1., 2., 3., invalid])
            with self.assertRaises(UnknownRejectionError):
                aggregate_windows(plan, np.zeros(4, dtype=int), np.ones((4, 2)), (("svm_margin", values),))
        with self.assertRaises(UnknownRejectionError):
            aggregate_windows(plan, np.array([0, 0, 0, 4]), np.ones((4, 2)), (("svm_margin", np.ones(4)),))
        with self.assertRaises(UnknownRejectionError):
            vote_identity(np.array([0, 1]), np.full((2, 2), np.nan))
