from dataclasses import replace
from pathlib import Path
from statistics import fmean, pstdev
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from face_recognition.application.evaluate_distance_novelty import DistanceNoveltyEvaluator, rank_novelty_results
from face_recognition.application.evaluate_rejection_scores import RejectionScoreEvaluator
from face_recognition.application.rejection_metrics import evaluate_threshold
from face_recognition.application.rejection_score_analysis import analyze_scores, score_distribution
from face_recognition.application.training_folds import create_session_folds
from face_recognition.application.unknown_rejection_split import UnknownRejectionError
from face_recognition.domain.models.distance_novelty import NOVELTY_MODEL, NOVELTY_NEIGHBORS, REFERENCE_SCORES
from face_recognition.domain.models.identity_split import IdentityPartition
from face_recognition.infrastructure.classifiers.sklearn_distance_novelty import SklearnDistanceNoveltyRunner
from face_recognition.infrastructure.classifiers.sklearn_feature_reduction import build_reduction_pipeline
from face_recognition.infrastructure.classifiers.sklearn_rejection_scores import SklearnRejectionScoreRunner
from face_recognition.infrastructure.features.known_distances import KnownDistanceScorer
from tests.unknown_rejection_fakes import rejection_fixture, load_rejection_fixture


class DistanceNoveltyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory(prefix="synthetic_distance_evaluation_")
        cls.processed, cls.source, cls.partition, _, _ = rejection_fixture(Path(cls.temporary.name))
        cls.known, cls.unknown = load_rejection_fixture(cls.processed, cls.source, cls.partition)
        cls.report = DistanceNoveltyEvaluator(SklearnDistanceNoveltyRunner()).evaluate(cls.known, cls.unknown)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_three_exact_neighbor_counts_two_references_two_folds_fixed_model(self):
        self.assertEqual(tuple((r.score_family, r.neighbor_count) for r in self.report.novelty_results), NOVELTY_NEIGHBORS)
        self.assertEqual(tuple(r.score_family for r in self.report.reference_results), REFERENCE_SCORES)
        self.assertEqual(self.report.model, NOVELTY_MODEL)
        self.assertEqual((NOVELTY_MODEL.k, NOVELTY_MODEL.C, NOVELTY_MODEL.gamma), (448, 4, .001))
        self.assertTrue(all(len(row.folds) == 2 for row in self.report.novelty_results + self.report.reference_results))

    def test_reference_metrics_and_closed_set_identity_are_unchanged(self):
        previous = RejectionScoreEvaluator(SklearnRejectionScoreRunner()).evaluate(self.known, self.unknown)
        for reference in self.report.reference_results:
            old = next(r for r in previous.results if r.candidate == NOVELTY_MODEL and r.score_family == reference.score_family)
            self.assertEqual(reference.mean_metrics, old.mean_metrics)
            self.assertEqual(reference.mean_closed_set_metrics, old.mean_closed_set_metrics)
            self.assertEqual(tuple(f.analysis for f in reference.folds), tuple(f.analysis for f in old.folds))
        self.assertTrue(all(r.mean_closed_set_metrics == self.report.reference_results[0].mean_closed_set_metrics
                            for r in self.report.novelty_results))

    def test_only_two_model_fits_reference_fit_and_preprocessing_use_known_fit_only(self):
        models, fit_inputs, references = [], [], []
        original_fit = KnownDistanceScorer.fit

        def build(*args):
            model = build_reduction_pipeline(*args)
            fit = model.fit

            def tracked_fit(X, y):
                fit_inputs.append((X.copy(), y.copy()))
                return fit(X, y)

            model.fit = tracked_fit
            models.append(model)
            return model

        def reference_fit(scorer, X, y, count):
            original_fit(scorer, X, y, count)
            references.append((scorer._reference.copy(), scorer._labels.copy()))
            return scorer

        with patch("face_recognition.infrastructure.classifiers.sklearn_unknown_rejection.build_reduction_pipeline", build), \
             patch.object(KnownDistanceScorer, "fit", reference_fit):
            DistanceNoveltyEvaluator(SklearnDistanceNoveltyRunner()).evaluate(self.known, self.unknown)
        self.assertEqual((len(models), len(fit_inputs), len(references)), (2, 2, 2))
        for fold, model, (X, y), (Z, zy) in zip(create_session_folds(self.known), models, fit_inputs, references):
            self.assertTrue(np.array_equal(X, fold.fit.X) and np.array_equal(y, fold.fit.y))
            self.assertTrue(np.allclose(Z, model[:-1].transform(fold.fit.X)) and np.array_equal(zy, fold.fit.y))
            self.assertEqual(model[0].k, 448)
            self.assertTrue(np.allclose(model[1].mean_, model[0].transform(X).mean(axis=0, dtype=np.float64)))
            self.assertEqual((model[-1].C, model[-1].gamma, model[-1].kernel, model[-1].probability), (4., .001, "rbf", False))

    def test_changed_validation_unknown_cannot_change_reference_or_model(self):
        fold = create_session_folds(self.known)[0]
        references, models = [], []
        original = KnownDistanceScorer.fit

        def fit(scorer, X, y, count):
            original(scorer, X, y, count)
            references.append((scorer._reference.copy(), scorer._labels.copy()))
            return scorer

        def build(*args):
            model = build_reduction_pipeline(*args)
            models.append(model)
            return model

        with patch.object(KnownDistanceScorer, "fit", fit), \
             patch("face_recognition.infrastructure.classifiers.sklearn_unknown_rejection.build_reduction_pipeline", build):
            runner = SklearnDistanceNoveltyRunner()
            runner.score_novelty_fold(*fold.fit.arrays(), *fold.validation.arrays(), self.unknown.X, 3)
            runner.score_novelty_fold(*fold.fit.arrays(), fold.validation.X + 100, fold.validation.y, self.unknown.X - 100, 3)
        self.assertTrue(np.array_equal(references[0][0], references[1][0]))
        self.assertTrue(np.array_equal(references[0][1], references[1][1]))
        self.assertTrue(np.array_equal(models[0][0].get_support(), models[1][0].get_support()))
        self.assertTrue(np.array_equal(models[0][1].mean_, models[1][1].mean_))
        self.assertTrue(np.array_equal(models[0][-1].support_vectors_, models[1][-1].support_vectors_))

    def test_identity_predictions_remain_svm_even_when_neighbor_labels_disagree(self):
        fold = create_session_folds(self.known)[0]
        runner = SklearnDistanceNoveltyRunner()
        original = runner.score_novelty_fold(*fold.fit.arrays(), *fold.validation.arrays(), self.unknown.X, 3)
        score = KnownDistanceScorer.score

        def wrong_neighbors(scorer, X):
            result = score(scorer, X)
            return replace(result, nearest_labels=(result.nearest_labels + 1) % 3)

        with patch.object(KnownDistanceScorer, "score", wrong_neighbors):
            altered = runner.score_novelty_fold(*fold.fit.arrays(), *fold.validation.arrays(), self.unknown.X, 3)
        self.assertTrue(np.array_equal(original.known_correct, altered.known_correct))
        self.assertEqual(original.closed_set_metrics, altered.closed_set_metrics)

    def test_nearest_identity_and_unknown_concentration_diagnostics_are_correct(self):
        models = []

        def build(*args):
            model = build_reduction_pipeline(*args)
            models.append(model)
            return model

        with patch("face_recognition.infrastructure.classifiers.sklearn_unknown_rejection.build_reduction_pipeline", build):
            report = DistanceNoveltyEvaluator(SklearnDistanceNoveltyRunner()).evaluate(self.known, self.unknown)
        for fold, model, diagnostic in zip(create_session_folds(self.known), models, report.neighbor_diagnostics):
            fit = model[:-1].transform(fold.fit.X).astype(np.float64)
            known = model[:-1].transform(fold.validation.X).astype(np.float64)
            unknown = model[:-1].transform(self.unknown.X).astype(np.float64)
            ky = fold.fit.y[np.argmin(np.linalg.norm(known[:, None] - fit[None, :], axis=2), axis=1)]
            uy = fold.fit.y[np.argmin(np.linalg.norm(unknown[:, None] - fit[None, :], axis=2), axis=1)]
            self.assertEqual(diagnostic.nearest_same_identity_rate, float((ky == fold.validation.y).mean()))
            self.assertEqual(diagnostic.unknown_nearest_majority_concentration, np.bincount(uy).max() / len(uy))
            self.assertEqual(diagnostic.reference_samples, len(fold.fit.indices))
        self.assertEqual(report.mean_nearest_same_identity_rate, fmean(d.nearest_same_identity_rate for d in report.neighbor_diagnostics))

    def test_aggregation_comparisons_and_positive_distance_distributions(self):
        for row in self.report.novelty_results:
            self.assertEqual(row.mean_roc_auc, fmean(f.analysis.roc_auc for f in row.folds))
            self.assertEqual(row.std_roc_auc, pstdev(f.analysis.roc_auc for f in row.folds))
            self.assertEqual(row.mean_eer, fmean(f.analysis.equal_error_point.eer for f in row.folds))
            self.assertEqual(row.std_eer, pstdev(f.analysis.equal_error_point.eer for f in row.folds))
            self.assertEqual(row.known_distance_distribution.count, 36)
            self.assertEqual(row.unknown_distance_distribution.count, 24)
            self.assertGreater(row.known_distance_distribution.mean, 0)
            thresholds = [f.analysis.calibration.threshold for f in row.folds]
            self.assertAlmostEqual(row.median_development_threshold, fmean(thresholds))
            self.assertEqual(row.absolute_threshold_difference, abs(thresholds[0] - thresholds[1]))
            for name in row.mean_metrics.__dataclass_fields__:
                self.assertAlmostEqual(getattr(row.mean_metrics, name), fmean(getattr(f.analysis.calibration.metrics, name) for f in row.folds))
                self.assertAlmostEqual(getattr(row.std_metrics, name), pstdev(getattr(f.analysis.calibration.metrics, name) for f in row.folds))
            for delta, reference in zip(row.vs_references, self.report.reference_results):
                self.assertEqual(delta.reference_family, reference.score_family)
                self.assertEqual(delta.roc_auc_delta, row.mean_roc_auc - reference.mean_roc_auc)
                self.assertEqual(delta.eer_delta, row.mean_eer - reference.mean_eer)
                self.assertEqual(delta.far_delta, row.mean_metrics.unknown_false_acceptance_rate - reference.mean_metrics.unknown_false_acceptance_rate)
                self.assertEqual(delta.frr_delta, row.mean_metrics.known_rejection_rate - reference.mean_metrics.known_rejection_rate)
                self.assertEqual(delta.known_correct_identification_delta, row.mean_metrics.known_correct_identification_rate - reference.mean_metrics.known_correct_identification_rate)
                self.assertEqual(delta.open_set_accuracy_delta, row.mean_metrics.open_set_exact_accuracy - reference.mean_metrics.open_set_exact_accuracy)

    def test_each_fold_and_score_has_own_analysis_and_far_points(self):
        calls = []

        def analyze(known, correct, unknown):
            calls.append((len(known), len(correct), len(unknown)))
            return analyze_scores(known, correct, unknown)

        with patch("face_recognition.application.evaluate_distance_novelty.analyze_scores", analyze):
            report = DistanceNoveltyEvaluator(SklearnDistanceNoveltyRunner()).evaluate(self.known, self.unknown)
        self.assertEqual(calls, [(18, 18, 12)] * 10)
        for row in report.novelty_results:
            for fold in row.folds:
                self.assertEqual(tuple(p.target_far for p in fold.analysis.far_operating_points), (.2, .1, .05))
                for point in fold.analysis.far_operating_points:
                    if point.metrics:
                        self.assertLessEqual(point.metrics.unknown_false_acceptance_rate, point.target_far)

    def test_protected_or_nonfinite_data_rejected_before_fitting(self):
        for kind in ("known_test", "unknown_holdout", "nan"):
            known, unknown = load_rejection_fixture(self.processed, self.source, self.partition)
            if kind == "known_test":
                held = next(r for r in self.source.samples if r.partition is IdentityPartition.KNOWN_TEST)
                object.__setattr__(known, "samples", (held, *known.samples[1:]))
            elif kind == "unknown_holdout":
                held = next(r for r in self.source.samples if r.person_id in self.partition.specification.holdout_ids)
                object.__setattr__(unknown, "samples", (held, *unknown.samples[1:]))
            else:
                unknown.X.setflags(write=True)
                unknown.X[0, 0] = np.nan
            runner = SklearnDistanceNoveltyRunner()
            with patch.object(runner, "score_novelty_fold") as score:
                with self.assertRaises(UnknownRejectionError):
                    DistanceNoveltyEvaluator(runner).evaluate(known, unknown)
                score.assert_not_called()

    def test_ranking_priorities_and_late_neighbor_tie_break(self):
        original = self.report.novelty_results[0]

        def row(family="mean_5nn", count=5, auc=.7, eer=.3, far=.3, correct=.4, accuracy=.5, frr=.5):
            return replace(original, score_family=family, neighbor_count=count, mean_roc_auc=auc, mean_eer=eer,
                           mean_metrics=replace(original.mean_metrics, unknown_false_acceptance_rate=far,
                                                known_correct_identification_rate=correct, open_set_exact_accuracy=accuracy,
                                                known_rejection_rate=frr))

        for better in (row(auc=.8), row(eer=.2), row(far=.2), row(correct=.5), row(accuracy=.6), row(frr=.4)):
            worse = row(family="nearest_1", count=1)
            self.assertEqual(rank_novelty_results((worse, better))[0], better.score_family)
        self.assertEqual(rank_novelty_results((row(), row(family="nearest_1", count=1)))[0], "nearest_1")
        self.assertEqual(rank_novelty_results(tuple(reversed(self.report.novelty_results))), self.report.ranking)
        self.assertEqual(self.report.best_distance_based_novelty_development_candidate, self.report.ranking[0])

    def test_known_metric_arithmetic_and_impractical_far_points(self):
        known, unknown, correct = np.array([-1., -2., -3., -4.]), np.array([-1.5, -2.5, -3.5, -4.5]), np.array([True, False, True, False])
        m = evaluate_threshold(known, correct, unknown, -2.5, allow_negative=True)
        self.assertEqual((m.unknown_false_acceptance_rate, m.known_rejection_rate, m.unknown_rejection_rate, m.known_acceptance_rate), (.5, .5, .5, .5))
        self.assertEqual((m.known_correct_identification_rate, m.known_misidentification_rate, m.open_set_exact_accuracy), (.25, .25, .375))
        self.assertEqual((m.binary_balanced_accuracy, m.binary_macro_f1), (.5, .5))
        analysis = analyze_scores(known, correct, unknown)
        self.assertEqual(analysis.roc_auc, .625)
        self.assertEqual(analysis.equal_error_point.eer, .5)
        impossible = analyze_scores(np.array([-10., -9.]), np.array([True, True]), np.array([-1., -2.]))
        self.assertTrue(all(p.status == "impractical" and p.metrics.known_acceptance_rate == 0 for p in impossible.far_operating_points))

    def test_positive_distance_summary_percentiles(self):
        values = np.array([1., 2., 3., 4., 5.])
        summary = score_distribution(values)
        self.assertEqual((summary.count, summary.mean, summary.median), (5, 3., 3.))
        self.assertAlmostEqual(summary.std, np.sqrt(2))
        self.assertEqual((summary.p10, summary.p25, summary.p75, summary.p90), (1.4, 2., 4., 4.6))
