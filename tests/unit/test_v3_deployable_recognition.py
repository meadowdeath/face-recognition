from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from face_recognition.application.train_v3_deployable_model import TrainV3DeployableModel
from face_recognition.application.v3_deployable_recognition import V3DeployableRecognizer
from face_recognition.application.v3_novelty_data import FrameScores
from face_recognition.domain.models.rejection import RejectionError
from face_recognition.domain.models.v3_deployable_model import IdentityDecision
from tests.v3_deployable_fakes import RecordingDeployableRunner, deployable_fixture


class V3DeployableRecognitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory(prefix="synthetic_v3_inference_")
        cls.data, cls.config, cls.provenance, *_ = deployable_fixture(Path(cls.temporary.name))
        cls.trained = TrainV3DeployableModel(RecordingDeployableRunner()).train(cls.data, cls.config, cls.provenance)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def service(self, model=None, runner=None):
        model = model or self.trained.model
        runner = runner or RecordingDeployableRunner()
        return V3DeployableRecognizer(model, runner, runner.fingerprints(model.fitted))

    def test_known_uses_svm_identity_and_equality_accepts_at_frozen_threshold(self):
        result = self.service().predict_window(self.data.X[:7])
        self.assertEqual(result.decision, IdentityDecision.KNOWN)
        self.assertEqual(result.identity, self.data.protocol.specification.known_ids[0])
        self.assertEqual(result.knownness_score, result.threshold)

    def test_unknown_suppresses_arbitrary_svm_known_identity(self):
        unknown = self.data.unknown(self.data.protocol.specification.unknown_ids)
        result = self.service().predict_window(unknown.X[:7])
        self.assertEqual(result.decision, IdentityDecision.UNKNOWN)
        self.assertIsNone(result.identity)
        self.assertLess(result.knownness_score, result.threshold)

    def test_window_median_and_existing_vote_sum_tie_rule(self):
        class Controlled(RecordingDeployableRunner):
            def infer(self, model, population):
                predicted = np.array([0, 0, 0, 1, 1, 1, 2], dtype=np.int32)
                decisions = np.array([[0., 2., 0.]] * 7)
                return FrameScores(predicted, decisions, (("mahalanobis_distance", np.array([-100., -5., -4., -3., -2., -1., 0.])),))
        # Pure inference fixture with three classes permits a 3/3/1 plurality tie.
        original = self.trained.model.fitted
        from types import SimpleNamespace
        fitted = SimpleNamespace(**{**vars(original), "identity_count": 3})
        model = replace(self.trained.model, fitted=fitted, class_order=("a", "b", "c"), threshold=-3.)
        result = self.service(model, Controlled()).predict_window(np.zeros((7, 1434), dtype=np.float32))
        self.assertEqual(result.knownness_score, -3.)
        self.assertEqual(result.identity, "b")
        self.assertEqual(result.decision, IdentityDecision.KNOWN)

    def test_fewer_frames_wrong_width_nonfinite_and_nonnumeric_inputs_fail(self):
        for values in (np.zeros((1, 1434)), np.zeros((7, 448)), np.full((7, 1434), np.nan), [["invalid"] * 1434] * 7):
            with self.assertRaises(RejectionError): self.service().predict_window(values)

    def test_refit_generation_or_changed_state_detected(self):
        runner = RecordingDeployableRunner()
        service = self.service(runner=runner)
        runner.fit_count += 1
        with self.assertRaises(RejectionError): service.predict_window(self.data.X[:7])
        service = self.service()
        model = self.trained.model.fitted
        before = model.offset
        try:
            model.offset += 1.
            with self.assertRaises(RejectionError): service.predict_window(self.data.X[:7])
        finally: model.offset = before
