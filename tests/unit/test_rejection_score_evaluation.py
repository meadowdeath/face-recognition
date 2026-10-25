from dataclasses import replace
from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from face_recognition.application.calibrate_unknown_rejection import UnknownRejectionCalibrator, two_fold_midpoint
from face_recognition.application.evaluate_rejection_scores import RejectionScoreEvaluator, rank_score_results
from face_recognition.application.training_folds import create_session_folds
from face_recognition.application.unknown_rejection_split import UnknownRejectionError
from face_recognition.domain.models.rejection_score_evaluation import SCORE_FAMILIES
from face_recognition.domain.models.identity_split import IdentityPartition
from face_recognition.domain.models.unknown_rejection import REJECTION_CANDIDATES
from face_recognition.infrastructure.classifiers.sklearn_rejection_scores import SklearnRejectionScoreRunner
from face_recognition.infrastructure.classifiers.sklearn_feature_reduction import build_reduction_pipeline
from face_recognition.infrastructure.classifiers.sklearn_unknown_rejection import SklearnUnknownRejectionRunner
from face_recognition.infrastructure.features.rejection_scores import PredictedClassCentroidScorer
from tests.unknown_rejection_fakes import rejection_fixture, load_rejection_fixture


class RejectionScoreEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory(prefix="synthetic_score_evaluation_")
        cls.processed, cls.source, cls.partition, _, _ = rejection_fixture(Path(cls.temporary.name))
        cls.known, cls.unknown = load_rejection_fixture(cls.processed, cls.source, cls.partition)
        cls.report = RejectionScoreEvaluator(SklearnRejectionScoreRunner()).evaluate(cls.known, cls.unknown)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_two_fixed_models_four_scores_two_folds_and_aggregation(self):
        self.assertEqual([(row.candidate, row.score_family) for row in self.report.results],
                         [(candidate, family) for candidate in REJECTION_CANDIDATES for family in SCORE_FAMILIES])
        for row in self.report.results:
            self.assertEqual(len(row.folds), 2)
            self.assertEqual(row.mean_roc_auc, fmean(fold.analysis.roc_auc for fold in row.folds))
            self.assertEqual(row.mean_eer, fmean(fold.analysis.equal_error_point.eer for fold in row.folds))
            self.assertEqual(row.median_development_threshold,
                             two_fold_midpoint(*(f.analysis.calibration.threshold for f in row.folds), allow_negative=True))
            self.assertEqual(row.pooled_known_distribution.count, 36)
            self.assertEqual(row.pooled_unknown_distribution.count, 24)
            for name in row.mean_metrics.__dataclass_fields__:
                self.assertAlmostEqual(getattr(row.mean_metrics, name),
                                       fmean(getattr(f.analysis.calibration.metrics, name) for f in row.folds))
            self.assertEqual(row.fold_gaps.roc_auc, abs(row.folds[0].analysis.roc_auc - row.folds[1].analysis.roc_auc))
            self.assertEqual(row.fold_gaps.far, abs(row.folds[0].analysis.calibration.metrics.unknown_false_acceptance_rate
                                                   - row.folds[1].analysis.calibration.metrics.unknown_false_acceptance_rate))

    def test_existing_margin_reference_and_closed_set_behavior_are_preserved(self):
        reference = UnknownRejectionCalibrator(SklearnUnknownRejectionRunner()).evaluate(self.known, self.unknown)
        for original in reference.results:
            rows = [row for row in self.report.results if row.candidate == original.candidate]
            margin = next(row for row in rows if row.score_family == "svm_margin")
            self.assertEqual(margin.mean_metrics, original.mean_metrics)
            self.assertEqual(tuple(f.analysis.calibration for f in margin.folds), tuple(f.calibration for f in original.folds))
            self.assertEqual(margin.mean_closed_set_metrics, original.mean_closed_set_metrics)
            self.assertTrue(all(row.mean_closed_set_metrics == original.mean_closed_set_metrics for row in rows))
            for row in rows:
                self.assertAlmostEqual(row.vs_margin.roc_auc_delta, row.mean_roc_auc - margin.mean_roc_auc)
                self.assertAlmostEqual(row.vs_margin.eer_delta, row.mean_eer - margin.mean_eer)
                self.assertAlmostEqual(row.vs_margin.far_delta,
                                       row.mean_metrics.unknown_false_acceptance_rate - margin.mean_metrics.unknown_false_acceptance_rate)

    def test_only_four_fits_centroids_and_scales_use_known_fit_data_only(self):
        models, references = [], []
        original_fit = PredictedClassCentroidScorer.fit

        def build(*args):
            model = build_reduction_pipeline(*args)
            models.append(model)
            return model

        def fit(scorer, X, y, identity_count):
            original_fit(scorer, X, y, identity_count)
            references.append((X.copy(), y.copy(), scorer._centroids.copy(), scorer._scales.copy()))
            return scorer

        with patch("face_recognition.infrastructure.classifiers.sklearn_unknown_rejection.build_reduction_pipeline", build), \
             patch.object(PredictedClassCentroidScorer, "fit", fit):
            RejectionScoreEvaluator(SklearnRejectionScoreRunner()).evaluate(self.known, self.unknown)
        self.assertEqual((len(models), len(references)), (4, 4))
        folds = create_session_folds(self.known)
        for index, (Z, labels, centers, scales) in enumerate(references):
            X, y = folds[index % 2].fit.arrays()
            self.assertTrue(np.allclose(Z, models[index][:-1].transform(X)))
            self.assertTrue(np.array_equal(labels, y))
            for label in range(3):
                self.assertTrue(np.allclose(centers[label], Z[labels == label].mean(axis=0, dtype=np.float64)))
                self.assertAlmostEqual(scales[label], np.median(np.linalg.norm(Z[labels == label] - centers[label], axis=1)))

    def test_validation_and_unknown_do_not_change_centroid_fit_statistics(self):
        fold = create_session_folds(self.known)[0]
        runner, references = SklearnRejectionScoreRunner(), []
        original = PredictedClassCentroidScorer.fit

        def fit(scorer, X, y, identities):
            original(scorer, X, y, identities)
            references.append((scorer._centroids.copy(), scorer._scales.copy()))
            return scorer

        with patch.object(PredictedClassCentroidScorer, "fit", fit):
            runner.score_fold_all(REJECTION_CANDIDATES[0], *fold.fit.arrays(), *fold.validation.arrays(), self.unknown.X, 3)
            runner.score_fold_all(REJECTION_CANDIDATES[0], *fold.fit.arrays(), fold.validation.X + 100, fold.validation.y,
                                  self.unknown.X - 100, 3)
        self.assertTrue(np.array_equal(references[0][0], references[1][0]))
        self.assertTrue(np.array_equal(references[0][1], references[1][1]))

    def test_ranking_requested_priorities_and_deterministic_final_tie(self):
        original = self.report.results[0]

        def row(family="svm_margin", candidate=REJECTION_CANDIDATES[0], auc=.7, eer=.3, far=.2, correct=.6,
                accuracy=.6, instability=.2):
            return replace(original, candidate=candidate, score_family=family, mean_roc_auc=auc, mean_eer=eer,
                           mean_metrics=replace(original.mean_metrics, unknown_false_acceptance_rate=far,
                                                known_correct_identification_rate=correct, open_set_exact_accuracy=accuracy),
                           absolute_threshold_difference=instability)

        pairs = ((row(auc=.8), row(family="svm_top1")), (row(eer=.2), row(family="svm_top1")),
                 (row(far=.1), row(family="svm_top1")), (row(correct=.7), row(family="svm_top1")),
                 (row(accuracy=.7), row(family="svm_top1")), (row(instability=.1), row(family="svm_top1")),
                 (row(candidate=REJECTION_CANDIDATES[1]), row(family="svm_top1")))
        for better, worse in pairs:
            self.assertEqual(rank_score_results((worse, better))[0], better.identifier)
        self.assertEqual(rank_score_results(tuple(reversed(self.report.results))), self.report.ranking)
        self.assertEqual(self.report.best_rejection_score_development_candidate, self.report.ranking[0])

    def test_protected_or_nonfinite_rows_fail_before_any_model_fit(self):
        for kind in ("known_test", "unknown_holdout", "nan"):
            known, unknown = load_rejection_fixture(self.processed, self.source, self.partition)
            if kind == "known_test":
                held = next(row for row in self.source.samples if row.partition is IdentityPartition.KNOWN_TEST)
                object.__setattr__(known, "samples", (held, *known.samples[1:]))
            elif kind == "unknown_holdout":
                held = next(row for row in self.source.samples if row.person_id in self.partition.specification.holdout_ids)
                object.__setattr__(unknown, "samples", (held, *unknown.samples[1:]))
            else:
                unknown.X.setflags(write=True)
                unknown.X[0, 0] = np.nan
            runner = SklearnRejectionScoreRunner()
            with patch.object(runner, "score_fold_all") as score:
                with self.assertRaises(UnknownRejectionError):
                    RejectionScoreEvaluator(runner).evaluate(known, unknown)
                score.assert_not_called()
