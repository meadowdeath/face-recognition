from copy import deepcopy
from dataclasses import replace
from math import inf, nextafter
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from face_recognition.application.refit_open_set_model import OpenSetModelRefitter
from face_recognition.application.rejection_score_analysis import analyze_scores
from face_recognition.application.rejection_metrics import calibrate_threshold
from face_recognition.application.temporal_windows import aggregate_windows, build_windows
from face_recognition.domain.models.identity_split import IdentityPartition
from face_recognition.domain.models.open_set_config import ConfigurationFreezeError
from face_recognition.domain.models.open_set_model import OpenSetModelError
from face_recognition.infrastructure.classifiers.sklearn_feature_reduction import build_reduction_pipeline
from face_recognition.infrastructure.classifiers.sklearn_open_set_refit import SklearnOpenSetRefitRunner
from face_recognition.infrastructure.features.known_distances import KnownDistanceScorer
from face_recognition.infrastructure.persistence.open_set_config_repository import read_frozen_configuration
from tests.open_set_refit_fakes import refit_fixture, load_refit_fixture


class OpenSetRefitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory(prefix="synthetic_refit_")
        cls.fixture = refit_fixture(Path(cls.temporary.name))
        cls.known, cls.unknown, cls.frozen = load_refit_fixture(cls.fixture)
        cls.result = OpenSetModelRefitter(SklearnOpenSetRefitRunner()).run(cls.frozen.configuration, cls.frozen.sha256, cls.known, cls.unknown)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_all_allowed_samples_dimension_and_deterministic_classes(self):
        summary, parts = self.result.summary, self.result.parts
        self.assertEqual((summary.known_training_samples, summary.identity_count, summary.selected_feature_count), (96, 3, 448))
        self.assertEqual(summary.original_feature_count, 1434)
        self.assertEqual(parts.class_ids, self.known.label_to_person)
        self.assertTrue(np.array_equal(parts.svm.classes_, np.arange(3)))
        reference, labels = parts.reference.reference_arrays()
        self.assertEqual(reference.shape, (96, 448))
        self.assertTrue(np.array_equal(labels, self.known.y))
        self.assertFalse(parts.svm.probability)
        self.assertFalse(summary.final_holdouts_evaluated or summary.known_test_payloads_accessed or summary.unknown_holdout_payloads_accessed)
        self.assertNotIn("array(", repr(self.result))

    def test_exactly_one_full_fit_and_known_only_reference_self_exclusion(self):
        fits, models, references, exclusions = [], [], [], []
        original_fit, original_score = KnownDistanceScorer.fit, KnownDistanceScorer.score

        def build(*args):
            model = build_reduction_pipeline(*args)
            original = model.fit

            def fit(X, y):
                fits.append((X.copy(), y.copy()))
                return original(X, y)

            model.fit = fit
            models.append(model)
            return model

        def fit_reference(scorer, X, y, count):
            references.append((X.copy(), y.copy()))
            return original_fit(scorer, X, y, count)

        def score(scorer, X, *, exclude_reference_indices=None):
            exclusions.append(None if exclude_reference_indices is None else exclude_reference_indices.copy())
            return original_score(scorer, X, exclude_reference_indices=exclude_reference_indices)

        with patch("face_recognition.infrastructure.classifiers.sklearn_open_set_refit.build_reduction_pipeline", build), \
             patch.object(KnownDistanceScorer, "fit", fit_reference), patch.object(KnownDistanceScorer, "score", score):
            OpenSetModelRefitter(SklearnOpenSetRefitRunner()).run(self.frozen.configuration, self.frozen.sha256, self.known, self.unknown)
        self.assertEqual((len(fits), len(references), len(models)), (1, 1, 1))
        X, y = fits[0]
        self.assertTrue(np.array_equal(X, self.known.X) and np.array_equal(y, self.known.y))
        self.assertTrue(np.allclose(references[0][0], models[0][:-1].transform(X)))
        self.assertTrue(np.array_equal(references[0][1], self.known.y))
        self.assertTrue(np.array_equal(exclusions[0], np.arange(96)))
        self.assertIsNone(exclusions[1])
        self.assertTrue(np.allclose(models[0][1].mean_, models[0][0].transform(X).mean(axis=0, dtype=np.float64)))

    def test_window_level_scores_match_self_excluded_reference_and_frozen_search(self):
        output = SklearnOpenSetRefitRunner().fit_and_score(self.frozen.configuration, self.known.X, self.known.y, self.unknown.X, self.known.label_to_person)
        kp, up = build_windows(self.known.samples, 7), build_windows(self.unknown.samples, 7)
        kw = aggregate_windows(kp, output.known.predicted, output.known.decisions, output.known.confidence)
        uw = aggregate_windows(up, output.unknown.predicted, output.unknown.decisions, output.unknown.confidence)
        labels = {person: index for index, person in enumerate(self.known.label_to_person)}
        correct = kw.predicted == np.array([labels[w.person_id] for w in kp.windows])
        expected = analyze_scores(kw.confidence[0][1], correct, uw.confidence[0][1])
        self.assertEqual(self.result.summary.analysis, expected)
        self.assertEqual((kp.counts.windows, kp.counts.used_samples, kp.counts.discarded_samples), (12, 84, 12))
        self.assertEqual((up.counts.windows, up.counts.used_samples, up.counts.discarded_samples), (4, 28, 4))
        self.assertEqual(expected.known_distribution.count, 12)
        self.assertEqual(expected.unknown_distribution.count, 4)
        for index, window in enumerate(kp.windows):
            self.assertEqual(kw.confidence[0][1][index], np.median(output.known.confidence[0][1][list(window.indices)]))

    def test_invalid_or_changed_frozen_json_rejected(self):
        path = self.fixture[-1]
        data = json.loads(path.read_text(encoding="utf-8"))
        for kind in ("incomplete", "mutable", "hyperparameter"):
            changed = deepcopy(data)
            if kind == "incomplete":
                del changed["architecture"]
            elif kind == "mutable":
                changed["status"] = "draft"
            else:
                changed["architecture"]["svm_parameters"]["C"] = 2.
            target = Path(self.temporary.name) / (kind + ".json")
            target.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaises(ConfigurationFreezeError):
                read_frozen_configuration(target)

    def test_protected_partition_and_wrong_dimensions_rejected_before_fit(self):
        for kind in ("known_test", "unknown_holdout", "dimensions"):
            known, unknown, frozen = load_refit_fixture(self.fixture)
            if kind == "known_test":
                held = next(s for s in self.fixture[1].samples if s.partition is IdentityPartition.KNOWN_TEST)
                object.__setattr__(known, "samples", (held, *known.samples[1:]))
            elif kind == "unknown_holdout":
                held = next(s for s in self.fixture[1].samples if s.person_id in self.fixture[2].specification.holdout_ids)
                object.__setattr__(unknown, "samples", (held, *unknown.samples[1:]))
            else:
                known, unknown = replace(known, X=known.X[:, :500]), replace(unknown, X=unknown.X[:, :500])
            runner = SklearnOpenSetRefitRunner()
            with patch.object(runner, "fit_and_score") as fit:
                with self.assertRaises(OpenSetModelError):
                    OpenSetModelRefitter(runner).run(frozen.configuration, frozen.sha256, known, unknown)
                fit.assert_not_called()

    def test_frozen_balanced_accuracy_ties_and_finite_threshold(self):
        known, unknown, correct = np.array([-1., -1.]), np.array([-1., -1.]), np.array([True, True])
        chosen = calibrate_threshold(known, correct, unknown, allow_negative=True)
        self.assertEqual(chosen.threshold, nextafter(-1., inf))
        self.assertEqual(chosen.metrics.binary_balanced_accuracy, .5)
        self.assertEqual(chosen.metrics.unknown_false_acceptance_rate, 0.)
        self.assertEqual(self.frozen.configuration.final_threshold_calibration_policy.selection_rule,
                         ("higher binary balanced accuracy", "lower FAR", "lower FRR", "higher threshold"))
        bad = replace(self.result.summary.analysis, calibration=replace(self.result.summary.analysis.calibration, threshold=float("nan")))
        with self.assertRaisesRegex(OpenSetModelError, "finite"):
            replace(self.result.summary, analysis=bad)
