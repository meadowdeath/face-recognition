import unittest
from unittest.mock import patch

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from face_recognition.application.training_folds import create_session_folds
from face_recognition.domain.models.feature_reduction import ReductionConfiguration
from face_recognition.infrastructure.classifiers.sklearn_baselines import BASELINE_CONFIGURATIONS, build_baseline_pipeline
from face_recognition.infrastructure.classifiers.sklearn_feature_reduction import build_reduction_pipeline
from face_recognition.infrastructure.features.feature_transformers import feature_transform_steps, fitted_reduction_metadata
from tests.feature_reduction_fakes import reduction_data


class FeatureTransformerTests(unittest.TestCase):
    def setUp(self):
        self.fold = create_session_folds(reduction_data())[0]
        self.X, self.y = self.fold.fit.arrays()
        self.validation = self.fold.validation.X

    def test_anova_fits_only_fit_rows_and_preserves_selected_coordinate_order(self):
        configuration = ReductionConfiguration("anova", k=4)
        model = build_reduction_pipeline(BASELINE_CONFIGURATIONS[0], configuration, 9)
        observed = []
        original = SelectKBest.fit

        def fit(selector, X, y=None):
            observed.append((X, y))
            return original(selector, X, y)

        with patch.object(SelectKBest, "fit", fit), threadpool_limits(limits=1):
            model.fit(self.X, self.y)
            model.predict(self.validation)
        self.assertEqual(len(observed), 1)
        self.assertTrue(np.shares_memory(observed[0][0], self.X))
        self.assertTrue(np.array_equal(observed[0][1], self.y), "Only fit labels reach ANOVA")
        expected_scores = f_classif(self.X, self.y)[0]
        self.assertTrue(np.allclose(model.named_steps["anova"].scores_, expected_scores))
        metadata = fitted_reduction_metadata(model, configuration, 9)
        self.assertEqual(metadata.retained_features, 4)
        indices = metadata.selected_feature_indices
        self.assertEqual(indices, tuple(sorted(indices)))
        self.assertTrue(np.array_equal(model.named_steps["anova"].transform(self.X), self.X[:, indices]))
        self.assertTrue(np.array_equal(model.named_steps["anova"].transform(self.validation), self.validation[:, indices]))

    def test_validation_changes_do_not_influence_scores_or_selected_indices(self):
        config = ReductionConfiguration("anova", k=3)
        first, second = (build_reduction_pipeline(BASELINE_CONFIGURATIONS[0], config, 9) for _ in range(2))
        with threadpool_limits(limits=1):
            first.fit(self.X, self.y)
            second.fit(self.X, self.y)
            first.predict(self.validation)
            second.predict(self.validation + 100)
        self.assertTrue(np.array_equal(first.named_steps["anova"].scores_, second.named_steps["anova"].scores_))
        self.assertEqual(fitted_reduction_metadata(first, config, 9), fitted_reduction_metadata(second, config, 9))

    def test_invalid_counts_and_variance_thresholds_are_explicitly_rejected(self):
        for args in (("anova", 0, None), ("anova", -1, None), ("anova", True, None),
                     ("pca", None, 0), ("pca", None, 1), ("pca", None, float("nan")),
                     ("pca", None, float("inf")), ("baseline", 2, None), ("other", None, None)):
            with self.subTest(case=args[:2]), self.assertRaises(ValueError):
                ReductionConfiguration(*args)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            feature_transform_steps(ReductionConfiguration("anova", k=10), 9)

    def test_requested_pipeline_order_and_unchanged_classifier_parameters(self):
        expected = {"baseline": (("scaler", "classifier"), ("classifier",)),
                    "anova": (("anova", "scaler", "classifier"), ("anova", "classifier")),
                    "pca": (("scaler", "pca", "classifier"), ("scaler", "pca", "classifier"))}
        configs = (ReductionConfiguration("baseline"), ReductionConfiguration("anova", k=2),
                   ReductionConfiguration("pca", explained_variance=.95))
        for reduction in configs:
            for classifier in BASELINE_CONFIGURATIONS:
                model = build_reduction_pipeline(classifier, reduction, 9)
                self.assertEqual(tuple(model.named_steps), expected[reduction.method][classifier.name == "Random Forest"])
                self.assertEqual(model.named_steps["classifier"].get_params(),
                                 build_baseline_pipeline(classifier).named_steps["classifier"].get_params())

    def test_pca_scaler_and_components_fit_only_fit_session(self):
        config = ReductionConfiguration("pca", explained_variance=.95)
        model = build_reduction_pipeline(BASELINE_CONFIGURATIONS[2], config, 9)
        original = PCA.fit_transform
        seen = []

        def fit_transform(pca, X, y=None):
            seen.append(X.copy())
            return original(pca, X, y)

        before = self.X.tobytes(), self.validation.tobytes()
        with patch.object(PCA, "fit_transform", fit_transform), threadpool_limits(limits=1):
            model.fit(self.X, self.y)
            model.predict(self.validation)
            model.predict(self.validation * 20)
        self.assertEqual(len(seen), 1)
        scaler = model.named_steps["scaler"]
        self.assertEqual(int(scaler.n_samples_seen_), len(self.y))
        self.assertTrue(np.allclose(scaler.mean_, self.X.mean(axis=0, dtype=np.float64)))
        self.assertTrue(np.allclose(seen[0], scaler.transform(self.X)))
        self.assertTrue(before == (self.X.tobytes(), self.validation.tobytes()), "Sources remain unchanged")
        metadata = fitted_reduction_metadata(model, config, 9)
        self.assertGreaterEqual(metadata.actual_explained_variance + 1e-6, .95)
        self.assertEqual(metadata.retained_features, model.named_steps["pca"].n_components_)
        self.assertEqual(model[:-1].transform(self.validation).shape[1], metadata.retained_features)
        # The immediately preceding dimension count falls short of the requested threshold.
        ratios = model.named_steps["pca"].explained_variance_ratio_
        if len(ratios) > 1:
            self.assertLess(float(ratios[:-1].sum()), .95)

    def test_pca_is_numerically_repeatable_and_validation_does_not_refit(self):
        config = ReductionConfiguration("pca", explained_variance=.9)
        models = [build_reduction_pipeline(BASELINE_CONFIGURATIONS[0], config, 9) for _ in range(2)]
        with threadpool_limits(limits=1):
            for model in models:
                model.fit(self.X, self.y)
            models[0].predict(self.validation)
            models[1].predict(self.validation - 100)
        self.assertTrue(np.allclose(models[0].named_steps["pca"].components_, models[1].named_steps["pca"].components_))
        self.assertTrue(np.allclose(models[0][:-1].transform(self.validation), models[1][:-1].transform(self.validation), atol=1e-5))

    def test_nonfinite_inputs_rejected_by_anova_and_pca(self):
        for reduction in (ReductionConfiguration("anova", k=2), ReductionConfiguration("pca", explained_variance=.9)):
            for value in (float("nan"), float("inf"), -float("inf")):
                X = self.X.copy()
                X[0, 0] = value
                with self.assertRaises(ValueError), threadpool_limits(limits=1):
                    build_reduction_pipeline(BASELINE_CONFIGURATIONS[0], reduction, 9).fit(X, self.y)
