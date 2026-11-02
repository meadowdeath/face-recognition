from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from face_recognition.application.rejection_metrics import decision_margin
from face_recognition.application.v3_novelty_data import V3Population
from face_recognition.domain.models.rejection import RejectionError
from face_recognition.infrastructure.classifiers.sklearn_v3_novelty import SklearnV3NoveltyRunner
from face_recognition.infrastructure.features.mahalanobis import PredictedClassMahalanobisScorer
from face_recognition.infrastructure.features.rejection_scores import PredictedClassCentroidScorer, decision_top1
from tests.v3_novelty_fakes import novelty_data


class V3NoveltyScoreTests(unittest.TestCase):
    def test_svm_knownness_definitions_and_common_orientation(self):
        decisions = np.array([[1., 4., 2.], [-5., -2., -4.], [3., 3., 1.]])
        np.testing.assert_array_equal(decision_top1(decisions), [4., -2., 3.])
        np.testing.assert_array_equal(decision_margin(decisions), [2., 2., 0.])
        for scorer in (PredictedClassCentroidScorer(), PredictedClassMahalanobisScorer()):
            scorer.fit(np.array([[0., 0.], [2., 0.], [10., 0.], [14., 0.]]), np.array([0, 0, 1, 1]), 2)
            scores = scorer.score(np.array([[1., 0.], [100., 0.]]), np.array([0, 0]))
            self.assertGreater(scores[0], scores[1])
            self.assertTrue(np.all(scores <= 0))

    def test_distance_queries_use_svm_predicted_class_not_nearest_class(self):
        X, y = np.array([[0., 0.], [2., 0.], [10., 0.], [14., 0.]]), np.array([0, 0, 1, 1])
        centroid = PredictedClassCentroidScorer().fit(X, y, 2)
        np.testing.assert_allclose(centroid.score(np.array([[2., 0.], [2., 0.]]), np.array([0, 1])), [-1., -10.])
        scorer = PredictedClassMahalanobisScorer().fit(X, y, 2)
        actual = scorer.score(np.array([[2., 0.]]), np.array([1]))[0]
        estimate = LedoitWolf(store_precision=False).fit(X[y == 1])
        cov = estimate.covariance_ + np.eye(2) * 1e-8 * max(1., np.trace(estimate.covariance_) / 2)
        delta = np.array([2., 0.]) - estimate.location_
        self.assertAlmostEqual(actual, -np.sqrt(delta @ np.linalg.solve(cov, delta)))

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

    def test_all_preprocessing_and_reference_fits_see_only_known_fit_rows(self):
        with TemporaryDirectory(prefix="synthetic_v3_fit_") as folder:
            data, _, _ = novelty_data(Path(folder), samples=7)
            fit = data.known((data.protocol.specification.known_sessions[0],))
            # This test uses real sklearn rather than the encoded fake runner;
            # perturb the synthetic role-marker columns to avoid constant scores.
            X = fit.X.copy()
            X[:, :2] += np.random.default_rng(4).normal(0., .01, X[:, :2].shape).astype(np.float32)
            fit = V3Population(X, fit.y, fit.samples)
            unknown = data.unknown(data.protocol.specification.unknown_ids)
            observed = {}
            covariance_rows = []
            targets = ((SelectKBest, "selector"), (StandardScaler, "scaler"), (SVC, "svm"),
                       (PredictedClassCentroidScorer, "centroid"), (PredictedClassMahalanobisScorer, "mahalanobis"))
            with ExitStack() as stack:
                original_covariance_fit = LedoitWolf.fit
                def fit_covariance(self, X, y=None):
                    covariance_rows.append(np.array(X))
                    return original_covariance_fit(self, X, y)
                stack.enter_context(patch.object(LedoitWolf, "fit", fit_covariance))
                for cls, name in targets:
                    original = cls.fit
                    def spy(self, X, y=None, *args, _original=original, _name=name, **kwargs):
                        observed[_name] = (np.array(X), None if y is None else np.array(y))
                        return _original(self, X, y, *args, **kwargs)
                    stack.enter_context(patch.object(cls, "fit", spy))
                runner = SklearnV3NoveltyRunner()
                model = runner.fit(fit, 2)
                result = runner.infer(model, unknown)
            self.assertEqual(set(observed), {name for _, name in targets})
            self.assertEqual(result.decisions.shape, (len(unknown.y), 2))
            for name, (X, y) in observed.items():
                self.assertEqual(len(X), len(fit.y))
                np.testing.assert_array_equal(y, fit.y)
                self.assertEqual(X.shape[1], 1434 if name == "selector" else 448)
            Z = model.pipeline[:-1].transform(fit.X)
            np.testing.assert_allclose(observed["centroid"][0], Z)
            np.testing.assert_allclose(observed["mahalanobis"][0], Z)
            self.assertEqual(len(covariance_rows), 2)
            for label, rows in enumerate(covariance_rows):
                np.testing.assert_allclose(rows, Z[fit.y == label])
            self.assertFalse(model.pipeline[-1].probability)
            self.assertEqual(model.pipeline[-1].C, 4.)
            self.assertEqual(model.pipeline[-1].gamma, .001)
            self.assertEqual(model.pipeline.named_steps["anova"].k, 448)
            with self.assertRaises(ValueError):
                runner.fit(unknown, 2)
