from dataclasses import replace
from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from face_recognition.application.calibrate_unknown_rejection import UnknownRejectionCalibrator, rank_rejection_candidates, two_fold_midpoint
from face_recognition.application.rejection_data import UnknownCalibrationData
from face_recognition.application.training_folds import create_session_folds
from face_recognition.application.unknown_rejection_split import UnknownRejectionError
from face_recognition.domain.models.identity_split import IdentityPartition
from face_recognition.domain.models.unknown_rejection import REJECTION_CANDIDATES
from face_recognition.infrastructure.classifiers.sklearn_feature_reduction import build_reduction_pipeline
from face_recognition.infrastructure.classifiers.sklearn_unknown_rejection import SklearnUnknownRejectionRunner, _margins
from face_recognition.infrastructure.persistence.unknown_calibration_loader import UnknownCalibrationLoader
from tests.unknown_rejection_fakes import load_rejection_fixture, rejection_fixture


class UnknownRejectionCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory(prefix="synthetic_rejection_calibration_")
        cls.processed, cls.source, cls.partition, _, _ = rejection_fixture(Path(cls.temporary.name))
        cls.known, cls.unknown = load_rejection_fixture(cls.processed, cls.source, cls.partition)
        cls.report = UnknownRejectionCalibrator(SklearnUnknownRejectionRunner()).evaluate(cls.known, cls.unknown)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_two_candidates_fold_threshold_aggregation_stability_and_closed_metrics(self):
        self.assertEqual(tuple(row.candidate for row in self.report.results), REJECTION_CANDIDATES)
        for row in self.report.results:
            thresholds = tuple(fold.calibration.threshold for fold in row.folds)
            self.assertEqual(row.absolute_threshold_difference, abs(thresholds[0] - thresholds[1]))
            self.assertEqual(row.development_rejection_threshold, two_fold_midpoint(*thresholds))
            self.assertEqual(row.mean_threshold, row.median_threshold)
            for name in row.mean_metrics.__dataclass_fields__:
                self.assertAlmostEqual(getattr(row.mean_metrics, name), fmean(getattr(f.calibration.metrics, name) for f in row.folds))
            for name in row.mean_closed_set_metrics.__dataclass_fields__:
                self.assertAlmostEqual(getattr(row.mean_closed_set_metrics, name), fmean(getattr(f.closed_set_metrics, name) for f in row.folds))
            for fold in row.folds:
                self.assertEqual((fold.fit_samples, fold.known_validation_samples, fold.unknown_calibration_samples), (18, 18, 12))
                self.assertFalse(set(fold.fit_session_ids) & set(fold.validation_session_ids))
                self.assertTrue(0 <= fold.at_development_threshold.open_set_exact_accuracy <= 1)
        self.assertEqual(self.report.selected_open_set_development_candidate.identifier, self.report.ranking[0])
        self.assertEqual(two_fold_midpoint(1, 4), 2.5)
        self.assertEqual(two_fold_midpoint(4, 1), 2.5)

    def test_selector_scaler_svm_fit_only_known_fit_rows_and_class_order_is_stable(self):
        folds = create_session_folds(self.known)
        observed, models = {"selector": [], "scaler": [], "svm": []}, []
        originals = SelectKBest.fit, StandardScaler.fit, SVC.fit

        def selector_fit(model, X, y=None):
            observed["selector"].append((X.copy(), y.copy()))
            return originals[0](model, X, y)

        def scaler_fit(model, X, y=None, **kwargs):
            observed["scaler"].append(X.copy())
            return originals[1](model, X, y, **kwargs)

        def svm_fit(model, X, y, **kwargs):
            observed["svm"].append((X.copy(), y.copy()))
            return originals[2](model, X, y, **kwargs)

        def builder(*args):
            model = build_reduction_pipeline(*args)
            models.append(model)
            return model

        before = self.known.X.tobytes(), self.unknown.X.tobytes()
        with patch.object(SelectKBest, "fit", selector_fit), patch.object(StandardScaler, "fit", scaler_fit), \
             patch.object(SVC, "fit", svm_fit), \
             patch("face_recognition.infrastructure.classifiers.sklearn_unknown_rejection.build_reduction_pipeline", builder):
            UnknownRejectionCalibrator(SklearnUnknownRejectionRunner()).evaluate(self.known, self.unknown)
        self.assertEqual([len(values) for values in observed.values()], [4, 4, 4])
        for index, model in enumerate(models):
            X, y = folds[index % 2].fit.arrays()
            self.assertTrue(np.array_equal(observed["selector"][index][0], X), "Only known fit rows reach ANOVA")
            self.assertTrue(np.array_equal(observed["selector"][index][1], y))
            self.assertTrue(np.array_equal(observed["scaler"][index], model.named_steps["anova"].transform(X)))
            self.assertTrue(np.allclose(observed["svm"][index][0], model[:-1].transform(X)))
            self.assertTrue(np.array_equal(observed["svm"][index][1], y))
            self.assertTrue(np.array_equal(model.classes_, np.arange(3)))
        self.assertTrue(before == (self.known.X.tobytes(), self.unknown.X.tobytes()))

    def test_unknown_and_known_validation_changes_do_not_influence_fitted_parameters(self):
        fold = create_session_folds(self.known)[0]
        scores = []
        original = SelectKBest.fit

        def selector_fit(model, X, y=None):
            result = original(model, X, y)
            scores.append(model.scores_.copy())
            return result

        runner = SklearnUnknownRejectionRunner()
        with patch.object(SelectKBest, "fit", selector_fit):
            first = runner.score_fold(REJECTION_CANDIDATES[0], *fold.fit.arrays(), *fold.validation.arrays(), self.unknown.X, 3)
            runner.score_fold(REJECTION_CANDIDATES[0], *fold.fit.arrays(), fold.validation.X + 100, fold.validation.y,
                              self.unknown.X - 100, 3)
        self.assertTrue(np.array_equal(scores[0], scores[1]))
        self.assertNotIn("array", repr(first))

    def test_binary_svc_expansion_is_fixed_and_invalid_class_columns_fail(self):
        class Binary:
            def decision_function(self, X):
                return np.array([-2., 1.])
        self.assertTrue(np.array_equal(_margins(Binary(), np.zeros((2, 4)), 2), np.array([4., 2.])))
        with self.assertRaises(ValueError):
            _margins(Binary(), np.zeros((2, 4)), 3)

    def test_ranking_all_requested_priorities_and_no_final_model_selection(self):
        original = self.report.results[0]

        def row(candidate, **values):
            metrics = replace(original.mean_metrics, open_set_exact_accuracy=.7, unknown_false_acceptance_rate=.2,
                              known_correct_identification_rate=.6, binary_balanced_accuracy=.7)
            return replace(original, candidate=candidate, absolute_threshold_difference=.1,
                           mean_metrics=replace(metrics, **values))

        a, b = REJECTION_CANDIDATES
        for field, better, worse in (("open_set_exact_accuracy", .8, .7), ("unknown_false_acceptance_rate", .1, .2),
                                    ("known_correct_identification_rate", .8, .7), ("binary_balanced_accuracy", .8, .7)):
            first, second = row(a, **{field: better}), row(b, **{field: worse})
            self.assertEqual(rank_rejection_candidates((second, first))[0], a.identifier)
        first, second = replace(row(a), absolute_threshold_difference=0), row(b)
        self.assertEqual(rank_rejection_candidates((second, first))[0], a.identifier)
        self.assertEqual(rank_rejection_candidates((row(a), row(b)))[0], b.identifier)
        self.assertEqual(rank_rejection_candidates(tuple(reversed(self.report.results))), self.report.ranking)

    def test_forged_protected_rows_and_nonfinite_calibration_rejected_before_scoring(self):
        for kind in ("unknown_holdout", "known_test", "nonfinite"):
            known, unknown = load_rejection_fixture(self.processed, self.source, self.partition)
            if kind == "unknown_holdout":
                held = next(row for row in self.source.samples if row.person_id == "unknown_hold_a")
                object.__setattr__(unknown, "samples", (held, *unknown.samples[1:]))
            elif kind == "known_test":
                held = next(row for row in self.source.samples if row.partition is IdentityPartition.KNOWN_TEST)
                object.__setattr__(known, "samples", (held, *known.samples[1:]))
            else:
                unknown.X.setflags(write=True)
                unknown.X[0, 0] = np.nan
            runner = SklearnUnknownRejectionRunner()
            with patch.object(runner, "score_fold") as score:
                with self.assertRaises(UnknownRejectionError):
                    UnknownRejectionCalibrator(runner).evaluate(known, unknown)
                score.assert_not_called()

    def test_loader_rejects_wrong_source_binding_dimensions_and_bad_calibration_payloads(self):
        loader = UnknownCalibrationLoader(self.processed)
        with self.assertRaises(ValueError):
            loader.load_calibration(self.source, replace(self.partition, source_identity_split_sha256="bad"), 513)
        with self.assertRaises(ValueError):
            loader.load_calibration(self.source, self.partition, 514)
        with self.assertRaises(ValueError):
            UnknownCalibrationData(self.unknown.X, tuple(reversed(self.unknown.samples)), self.source, self.partition)
