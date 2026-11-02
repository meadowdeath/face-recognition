from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import warnings

from face_recognition.application.analyze_v3_mahalanobis_diagnostics import V3MahalanobisDiagnosticAnalyzer
from face_recognition.domain.models.rejection import RejectionError
from face_recognition.infrastructure.classifiers.sklearn_v3_same_model import SklearnV3SameModelRunner
from face_recognition.infrastructure.persistence.v3_mahalanobis_diagnostics_report import mahalanobis_diagnostics_to_json
from tests.v3_mahalanobis_diagnostics_fakes import DiagnosticReplayRunner, diagnostic_fixture
from tests.v3_mahalanobis_fakes import RecordingSameModelRunner


class MahalanobisDiagnosticReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory(prefix="synthetic_v3_diagnostic_replay_")
        cls.data, _, _, _, cls.evidence, cls.provenance = diagnostic_fixture(Path(cls.temporary.name))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def analyze(self, runner=None, evidence=None, provenance=None, data=None):
        return V3MahalanobisDiagnosticAnalyzer(runner or DiagnosticReplayRunner()).analyze(
            data or self.data, evidence or self.evidence, provenance or self.provenance)

    def test_replay_uses_original_fit_sessions_saved_thresholds_and_only_outer_observations(self):
        runner = DiagnosticReplayRunner()
        result = self.analyze(runner)
        self.assertEqual(runner.fit_count, 4)
        self.assertEqual(len(runner.model_calls), 8)  # No calibration inference.
        fits = [e for e in runner.events if e[0] == "fit"]
        for i, (saved, row) in enumerate(zip(self.evidence.outer_results, result.folds)):
            self.assertEqual(row.saved_threshold, saved.threshold)
            self.assertTrue(row.saved_model_state_verified and row.saved_outer_summaries_verified)
            self.assertEqual(fits[i][1:], (saved.roles.known_fit_sessions, self.data.protocol.specification.known_ids,
                                         saved.fit_known_frames, ("mahalanobis_distance",)))
            known_call, unknown_call = runner.model_calls[2 * i:2 * i + 2]
            self.assertEqual(known_call[1], unknown_call[1])
            self.assertEqual(known_call[2:], (self.data.protocol.specification.known_ids, (saved.roles.known_test_session,)))
            self.assertEqual(unknown_call[2], saved.roles.unknown_test_ids)
            self.assertFalse(set(unknown_call[2]) & set(saved.roles.unknown_calibration_ids))
            self.assertEqual(row.known_counts, saved.known_test_counts)
            self.assertEqual(row.unknown_counts, saved.unknown_test_counts)
        self.assertEqual(result.aggregate.known_windows.total_windows, 104)
        self.assertEqual(result.aggregate.unknown_windows.total_windows, 208)
        self.assertEqual(len(result.aggregate.known_condition_by_session), 4)
        self.assertEqual(len(result.aggregate.unknown_identity_severity), 8)

    def test_no_threshold_search_selection_or_diagnostic_cutpoint_enumeration(self):
        forbidden = (
            "face_recognition.application.rejection_metrics.calibrate_threshold",
            "face_recognition.application.rejection_metrics.threshold_points",
            "face_recognition.application.evaluate_v3_mahalanobis_calibration.calibrate_threshold",
            "face_recognition.application.evaluate_v3_mahalanobis_calibration.evaluate_pair",
        )
        with ExitStack() as stack:
            for name in forbidden:
                stack.enter_context(patch(name, side_effect=AssertionError("No search/calibration in diagnostics")))
            result = self.analyze()
        self.assertEqual([f.saved_threshold for f in result.folds], [f.threshold for f in self.evidence.outer_results])

    def test_saved_model_digest_mismatch_rejected_before_any_scoring(self):
        row = self.evidence.outer_results[0]
        changed = replace(row, model_sha256_before_calibration="0" * 64, model_sha256_after_test="0" * 64)
        runner = DiagnosticReplayRunner()
        with self.assertRaisesRegex(RejectionError, "Reconstructed model differs"):
            self.analyze(runner, replace(self.evidence, outer_results=(changed, *self.evidence.outer_results[1:])))
        self.assertEqual(runner.model_calls, [])

    def test_inference_difference_cannot_be_reported_as_original_study(self):
        # The fake learned state is identical, but this adapter emits different
        # observations. Saved counts/metrics/distributions must also match.
        with self.assertRaisesRegex(RejectionError, "outer summaries differ"):
            self.analyze(RecordingSameModelRunner())

    def test_saved_score_distribution_and_operational_metric_mismatch_rejected(self):
        row = self.evidence.outer_results[0]
        altered_distribution = replace(row.diagnostics.known_distribution, mean=row.diagnostics.known_distribution.mean + 1.)
        changes = (
            replace(row, diagnostics=replace(row.diagnostics, known_distribution=altered_distribution)),
            replace(row, operational_metrics=replace(row.operational_metrics, known_rejection_rate=.999)),
            replace(row, threshold=row.threshold + 100.),
        )
        for changed in changes:
            with self.subTest(changed=changed.threshold), self.assertRaises(RejectionError):
                self.analyze(evidence=replace(self.evidence, outer_results=(changed, *self.evidence.outer_results[1:])))

    def test_model_mutation_refit_or_identity_replacement_during_inference_rejected(self):
        for defect in ("state", "refit", "identity"):
            class BadRunner(DiagnosticReplayRunner):
                identity_generation = 0
                def infer(self, model, population):
                    frames = super().infer(model, population)
                    if defect == "state": model.offset += 1.
                    elif defect == "refit": self.fit_count += 1
                    else: self.identity_generation += 1
                    return frames
                def model_identity(self, model):
                    return (id(model), self.identity_generation)
            with self.subTest(defect=defect), self.assertRaisesRegex(RejectionError, "changed during diagnostics"):
                self.analyze(BadRunner())

    def test_environment_protocol_payload_and_reference_provenance_mismatch_rejected_before_fit(self):
        for evidence in (replace(self.evidence, protocol_sha256="0" * 64), replace(self.evidence, payload_sha256="0" * 64),
                         replace(self.evidence, versions=replace(self.evidence.versions, scikit_learn="incompatible"))):
            runner = DiagnosticReplayRunner()
            with self.assertRaises(RejectionError): self.analyze(runner, evidence)
            self.assertEqual(runner.fit_count, 0)
        for provenance in (replace(self.provenance, evaluation_report_sha256="bad"),
                           replace(self.provenance, catalog_sha256="0" * 64),
                           replace(self.provenance, protocol_reference="C:\\private\\protocol.json"),
                           replace(self.provenance, evaluation_report_reference="../evaluation.json")):
            runner = DiagnosticReplayRunner()
            with self.assertRaises(RejectionError): self.analyze(runner, provenance=provenance)
            self.assertEqual(runner.fit_count, 0)

    def test_repeated_inputs_are_byte_identical_aggregate_only_with_explicit_methodology(self):
        first = self.analyze()
        self.assertEqual(mahalanobis_diagnostics_to_json(first), mahalanobis_diagnostics_to_json(self.analyze()))
        self.assertEqual(first.threshold_transport, "none_same_model_calibration")
        self.assertIn("not extracted", first.replay_limitation)
        self.assertIn("V2 remains the last independent external holdout", first.methodology)
        self.assertIn("Never pool raw scores", first.aggregation_policy)

    def test_real_sklearn_learned_state_and_saved_summaries_reproduce_without_calibration(self):
        with TemporaryDirectory(prefix="synthetic_real_diagnostic_replay_") as directory, warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            data, _, _, _, source, provenance = diagnostic_fixture(Path(directory), SklearnV3SameModelRunner())
            with patch("face_recognition.application.rejection_metrics.calibrate_threshold", side_effect=AssertionError("No calibration")), \
                    patch("face_recognition.application.rejection_metrics.threshold_points", side_effect=AssertionError("No enumeration")):
                first = self.analyze(SklearnV3SameModelRunner(), source, provenance, data)
                second = self.analyze(SklearnV3SameModelRunner(), source, provenance, data)
            self.assertEqual(mahalanobis_diagnostics_to_json(first), mahalanobis_diagnostics_to_json(second))
