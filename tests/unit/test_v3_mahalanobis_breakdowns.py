import unittest

import numpy as np

from face_recognition.application.v3_mahalanobis_breakdowns import (
    aggregate_breakdowns, fold_breakdowns, interpret_breakdowns, known_summary, severity, unknown_summary,
)
from face_recognition.application.v3_novelty_data import WindowBatch
from face_recognition.domain.models.rejection import RejectionError
from face_recognition.domain.models.temporal import WindowCounts


def batch(keys, scores, correct=None):
    return WindowBatch(tuple((*key, tuple(range(7))) for key in keys), np.array(scores, dtype=float),
                       np.array(correct if correct is not None else [False] * len(scores), dtype=bool),
                       WindowCounts(len(scores) * 7, len(scores) * 7, 0, len(scores)))


class MahalanobisBreakdownTests(unittest.TestCase):
    def setUp(self):
        known = batch((("k_a", "visit_a", "center"), ("k_b", "visit_a", "center"),
                       ("k_a", "visit_a", "yaw_left"), ("k_b", "visit_a", "yaw_left"), ("k_a", "visit_a", "near")),
                       [-1, 2, 1, 3, -2], [False, True, False, True, True])
        unknown = batch((("u_a", "x", "center"), ("u_a", "y", "far"), ("u_b", "x", "center")), [1, 2, -1])
        self.fold = fold_breakdowns("outer_a", 0., known, unknown, "visit_a", ("u_a", "u_b"))

    def test_known_condition_counts_correct_acceptance_misidentification_and_scores(self):
        rows = {r.label: r for r in self.fold.known_per_condition}
        center = rows["center"]
        self.assertEqual((center.counts.total_windows, center.counts.accepted_windows, center.counts.rejected_windows), (2, 1, 1))
        self.assertEqual((center.counts.frr, center.counts.correct_identification_rate, center.counts.accepted_misidentification_rate), (.5, .5, 0.))
        self.assertEqual((center.scores.mean, center.scores.median), (.5, .5))
        yaw = rows["yaw_left"]
        self.assertEqual((yaw.counts.correct_identification_windows, yaw.counts.accepted_misidentification_windows), (1, 1))
        self.assertEqual((yaw.counts.frr, yaw.counts.accepted_misidentification_rate), (0., .5))
        self.assertEqual(rows["near"].counts.frr, 1.)
        self.assertEqual(rows["pitch_up"].counts.total_windows, 0)
        self.assertIsNone(rows["pitch_up"].counts.frr)
        self.assertIsNone(rows["pitch_up"].scores.mean)

    def test_unknown_condition_and_identity_counts_far_rejection_and_scores(self):
        conditions = {r.label: r for r in self.fold.unknown_per_condition}
        self.assertEqual((conditions["center"].counts.accepted_windows, conditions["center"].counts.rejected_windows), (1, 1))
        self.assertEqual((conditions["center"].counts.far, conditions["center"].counts.unknown_rejection_rate), (.5, .5))
        self.assertEqual(conditions["far"].counts.far, 1.)
        identities = {r.label: r for r in self.fold.unknown_per_identity}
        self.assertEqual(identities["u_a"].counts.total_windows, 2)
        self.assertEqual(identities["u_a"].counts.far, 1.)
        self.assertEqual((identities["u_a"].scores.mean, identities["u_a"].scores.median), (1.5, 1.5))
        self.assertEqual(identities["u_b"].counts.far, 0.)

    def test_known_session_summary_and_condition_groups(self):
        session = self.fold.known_per_session[0]
        self.assertEqual(session.label, "visit_a")
        self.assertEqual((session.counts.total_windows, session.counts.accepted_windows, session.counts.rejected_windows), (5, 3, 2))
        self.assertEqual(session.counts.frr, .4)
        self.assertEqual(session.counts.correct_identification_rate, .4)
        self.assertEqual(session.counts.accepted_misidentification_rate, .2)
        self.assertEqual((session.scores.mean, session.scores.median), (.6, 1.))
        groups = {r.label: r for r in self.fold.known_condition_groups}
        self.assertEqual(groups["pose"].counts.total_windows, 2)
        self.assertEqual(groups["distance"].counts.frr, 1.)
        self.assertIsNone(groups["translation"].counts.frr)

    def test_exact_score_statistics_equality_accepts_and_nonfinite_rejected(self):
        row = known_summary("all", np.array([-3., -1., 0., 1.]), np.array([True, False, True, False]), 0.)
        self.assertEqual((row.counts.accepted_windows, row.counts.rejected_windows), (2, 2))
        self.assertEqual(row.counts.accepted_misidentification_rate, .25)
        self.assertEqual((row.scores.mean, row.scores.median), (-.75, -.5))
        for invalid in (np.nan, np.inf):
            with self.assertRaises(RejectionError): unknown_summary("all", np.array([invalid]), 0.)

    def test_contribution_ranking_and_all_largest_ties_are_deterministic(self):
        self.assertEqual(self.fold.largest_known_rejection_conditions, ("center", "near"))
        self.assertEqual(self.fold.largest_unknown_false_acceptance_identities, ("u_a",))
        top = self.fold.known_rejection_contributions[0]
        self.assertEqual((top.label, top.fraction_of_population_errors, top.fraction_of_population_windows), ("center", .5, .2))
        self.assertEqual(self.fold.unknown_false_acceptance_contributions[0].fraction_of_population_errors, 1.)

    def test_severity_uses_rate_then_count_then_lexical_label(self):
        rows = (unknown_summary("b", np.array([1., -1.]), 0.),
                unknown_summary("a", np.array([1., -1.]), 0.),
                unknown_summary("more_errors", np.array([1., 1., -1., -1.]), 0.),
                unknown_summary("higher_rate", np.array([1.]), 0.),
                unknown_summary("empty", np.array([]), 0.))
        self.assertEqual([r.label for r in severity(rows, known=False)], ["higher_rate", "more_errors", "a", "b", "empty"])
        self.assertEqual(severity(rows, known=False), severity(reversed(rows), known=False))

    def test_no_errors_has_no_largest_contributor_or_undefined_error_fraction_claim(self):
        known = batch((("k", "visit", "center"),), [1.], [True])
        unknown = batch((("u", "x", "center"),), [-1.])
        fold = fold_breakdowns("outer", 0., known, unknown, "visit", ("u",))
        self.assertEqual(fold.largest_known_rejection_conditions, ())
        self.assertEqual(fold.largest_unknown_false_acceptance_identities, ())
        self.assertTrue(all(r.fraction_of_population_errors is None for r in fold.known_rejection_contributions))
        self.assertIn("No known-window rejections", " ".join(interpret_breakdowns(aggregate_breakdowns((fold,)))))
        # Grouping alone cannot attest to evidence reproduction.
        self.assertFalse(fold.saved_model_state_verified)

    def test_aggregate_counts_sessions_conditions_and_scores_do_not_pool_raw_fold_scales(self):
        known = batch(tuple((f"k_{i}", "visit_b", "center") for i in range(4)), [100, 101, 102, 103], [True] * 4)
        unknown = batch((("u_c", "x", "center"), ("u_d", "x", "center")), [101, 99])
        second = fold_breakdowns("outer_b", 100., known, unknown, "visit_b", ("u_c", "u_d"))
        aggregate = aggregate_breakdowns((self.fold, second))
        center = next(r for r in aggregate.known_condition_severity if r.label == "center")
        self.assertEqual((center.counts.total_windows, center.counts.rejected_windows), (6, 1))
        self.assertEqual(center.counts.frr, 1 / 6)
        self.assertEqual(center.scores.mean_of_fold_means, 51.)  # (0.5 + 101.5)/2, not raw pooled mean.
        self.assertEqual(center.scores.std_of_fold_means, 50.5)
        self.assertEqual(center.scores.mean_of_fold_medians, 51.)
        self.assertEqual(aggregate.known_condition_severity[0].label, "near")
        self.assertEqual([r.label for r in aggregate.known_session_severity], ["visit_a", "visit_b"])
        self.assertEqual([r.session_id for r in aggregate.known_condition_by_session], ["visit_a", "visit_b"])
        self.assertEqual(aggregate.known_windows.total_windows, 9)
        self.assertEqual(aggregate.unknown_windows.accepted_windows, 3)
        text = " ".join(interpret_breakdowns(aggregate))
        self.assertIn("no model selection or calibration decisions changed", text)
        self.assertIn("not a causal", text)
        self.assertIn("no lighting labels", text.lower())
        with self.assertRaises(RejectionError): aggregate_breakdowns((self.fold, self.fold))
