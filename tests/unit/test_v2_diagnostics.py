from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, replace
from hashlib import sha256
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from face_recognition.application.evaluate_v2_open_set_diagnostics import V2DiagnosticEvaluator, _known_diagnostic, _windows
from face_recognition.application.temporal_windows import vote_identity
from face_recognition.application.v2_rejection import rank_key
from face_recognition.application.v2_rejection import calibrate as real_calibrate
from face_recognition.application.v2_study_data import V2StudyError, study_folds
from face_recognition.infrastructure.classifiers.sklearn_temporal_open_set import FrameInference
from face_recognition.infrastructure.classifiers.sklearn_v2_diagnostics import SklearnV2DiagnosticRunner, identity_summary
from face_recognition.infrastructure.features.known_distances import KnownDistanceScorer
from face_recognition.infrastructure.features.rejection_scores import PredictedClassCentroidScorer, decision_top1
from face_recognition.infrastructure.persistence.baseline_evaluation_report import BaselineReportError
from face_recognition.infrastructure.persistence.v2_diagnostics_report import v2_report_to_json, write_v2_report
from face_recognition.infrastructure.persistence.v2_study_loader import V2StudyLoader, read_v1_reference
from face_recognition.presentation.cli.evaluate_v2_open_set_diagnostics import main
from tests.v2_diagnostic_fakes import study_fixture


class V2DiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory(prefix="synthetic_v2_")
        cls.directory = Path(cls.temporary.name)
        cls.processed, cls.source, cls.partition, cls.source_path, cls.partition_path, cls.reference = study_fixture(cls.directory)
        cls.baseline = read_v1_reference(cls.reference, cls.source, cls.partition)
        cls.data = V2StudyLoader(cls.processed).load_study(cls.source, cls.partition, cls.baseline)
        cls.before = {p: (sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
                      for p in cls.directory.rglob("*") if p.is_file()}
        cls.fit_calls = []
        cls.fold_inputs = []
        original = SklearnV2DiagnosticRunner.infer_fold

        def infer(runner, X, y, K, ky, U, n):
            held_session = study_folds(cls.source)[len(cls.fold_inputs)].calibration_session
            held = np.array([s.session_id == held_session for s in cls.data.known_samples])
            cls.fold_inputs.append((bool(np.array_equal(X, cls.data.known_X[~held])),
                                    bool(np.array_equal(K, cls.data.known_X[held])),
                                    bool(np.array_equal(U, cls.data.unknown_X))))
            return original(runner, X, y, K, ky, U, n)

        def spy(model_type):
            original_fit = model_type.fit

            def fit(instance, X, y=None, **kwargs):
                cls.fit_calls.append((model_type.__name__, len(X), X.shape[1]))
                if model_type in (KnownDistanceScorer, PredictedClassCentroidScorer):
                    return original_fit(instance, X, y, kwargs.pop("identity_count", 3), **kwargs)
                return original_fit(instance, X, y, **kwargs)
            return fit

        # Neighbor/centroid have a positional identity_count, so use explicit spies.
        old_nn, old_centroid = KnownDistanceScorer.fit, PredictedClassCentroidScorer.fit

        def nn(instance, X, y, n):
            cls.fit_calls.append(("KnownDistanceScorer", len(X), X.shape[1]))
            return old_nn(instance, X, y, n)

        def centroid(instance, X, y, n):
            cls.fit_calls.append(("PredictedClassCentroidScorer", len(X), X.shape[1]))
            return old_centroid(instance, X, y, n)

        with patch.object(SklearnV2DiagnosticRunner, "infer_fold", infer), patch.object(SelectKBest, "fit", spy(SelectKBest)), \
                patch.object(StandardScaler, "fit", spy(StandardScaler)), patch.object(SVC, "fit", spy(SVC)), \
                patch.object(KnownDistanceScorer, "fit", nn), patch.object(PredictedClassCentroidScorer, "fit", centroid):
            cls.report = V2DiagnosticEvaluator().evaluate(cls.data, cls.baseline, SklearnV2DiagnosticRunner())

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_three_session_folds_exact_oof_coverage(self):
        folds = study_folds(self.source)
        self.assertEqual([f.calibration_session for f in folds], ["reserved_visit", "visit_b", "visit_a"])
        for f in folds:
            self.assertNotIn(f.calibration_session, f.fit_sessions)
            self.assertEqual(len(f.fit_sessions), 2)
        self.assertTrue(all(all(result) for result in self.fold_inputs))
        self.assertEqual(len(self.fold_inputs), 3)
        self.assertEqual(self.report.closed_set_frame_oof.samples, len(self.data.known_samples))
        self.assertEqual(sum(sum(row) for row in self.report.closed_set_frame_oof.confusion_matrix), len(self.data.known_samples))

    def test_all_learned_objects_fit_known_fit_only(self):
        for kind in ("SelectKBest", "StandardScaler", "SVC", "KnownDistanceScorer", "PredictedClassCentroidScorer"):
            calls = [r for r in self.fit_calls if r[0] == kind]
            self.assertEqual(len(calls), 3)
            self.assertTrue(all(n == 96 for _, n, _ in calls))
        self.assertTrue(all(width == 448 for name, _, width in self.fit_calls if name != "SelectKBest"))

    def test_exact_configurations_rotations_and_folds(self):
        self.assertEqual(len(self.report.candidates), 42)
        for r in self.report.candidates:
            self.assertEqual(len(r["cells"]), 9)
            self.assertEqual({c["fold"] for c in r["cells"]}, {"Fold A", "Fold B", "Fold C"})
            self.assertEqual({c["held_unknown_rotation"] for c in r["cells"]}, {1, 2, 3})
            self.assertTrue(all(c["unknown_calibration_windows"] > 0 and c["unknown_evaluation_windows"] > 0 for c in r["cells"]))

    def test_exact_confusion_and_repetition_accounting(self):
        for row in self.report.candidates:
            open_cm = np.array(row["open_set_confusion"]["observation_counts"])
            binary = np.array(row["binary_confusion"]["observation_counts"])
            adjusted = np.array(row["open_set_confusion"]["role_adjusted_counts"])
            self.assertEqual(int(open_cm.sum()), int(binary.sum()))
            self.assertEqual(int(open_cm[:-1].sum()), sum(d["window_observations"] for d in row["known_identity_diagnostics"]))
            self.assertTrue(bool(np.allclose(adjusted[:-1], open_cm[:-1] / 3)))
            self.assertTrue(bool(np.array_equal(adjusted[-1], open_cm[-1])))
            metrics = row["metrics"]
            self.assertAlmostEqual(metrics["known_rejection_rate"], open_cm[:-1, -1].sum() / open_cm[:-1].sum())
            self.assertAlmostEqual(metrics["unknown_false_acceptance_rate"], open_cm[-1, :-1].sum() / open_cm[-1].sum())
            self.assertAlmostEqual(metrics["open_set_exact_accuracy"], np.trace(adjusted) / adjusted.sum())
            self.assertEqual(row["open_set_confusion"]["labels"], self.source.specification.known_ids + ("UNKNOWN",))

    def test_person_condition_counts_and_diagnostics(self):
        for row in self.report.candidates:
            self.assertEqual(sum(r["source_samples"] for r in row["known_identity_diagnostics"]), len(self.data.known_samples))
            self.assertEqual(sum(r["source_samples"] for r in row["known_condition_diagnostics"]), len(self.data.known_samples))
            self.assertEqual(sum(r["unique_source_samples"] for r in row["unknown_condition_diagnostics"]), len(self.data.unknown_samples))
            self.assertTrue(all("person_id" not in r for r in row["unknown_condition_diagnostics"]))
            for person in row["known_identity_diagnostics"]:
                self.assertEqual(person["window_observations"], person["unique_windows"] * 3)
                self.assertAlmostEqual(person["known_acceptance_rate"] + person["known_rejection_rate"], 1)

    def test_window_counts_discard_efficiency_and_votes(self):
        for row in self.report.candidates:
            size = row["configuration"]["window"]
            for fold in row["window_counts"]:
                k = fold["known_unique_within_fold"]
                self.assertEqual(k["used_samples"] + k["discarded_samples"], k["source_samples"])
                self.assertEqual(k["windows"] * size, k["used_samples"])
            for vote in row["votes"]:
                rates = vote["known"]["rates"]
                self.assertAlmostEqual(sum(rates.values()), 1)
                self.assertTrue(0 < vote["known_mean_agreement"] <= 1)

    def test_threshold_support_and_fallback_reported(self):
        for row in self.report.candidates:
            for cell in row["cells"]:
                thresholds = cell["identity_thresholds"]
                if row["configuration"]["per_identity"]:
                    self.assertEqual(len(thresholds), 3)
                    for t in thresholds:
                        if not t["available"]:
                            self.assertIsNone(t["threshold"])
                            self.assertEqual(t["used_threshold"], cell["global_threshold"])
                        else:
                            self.assertGreaterEqual(t["eligible_known_support"], 10)
                            self.assertGreaterEqual(t["eligible_unknown_support"], 10)
                else:
                    self.assertEqual(len(cell["far_constrained_calibration_points"]), 3)

    def test_distributions_only_aggregate_and_empty_groups_defined(self):
        self.assertEqual(len(self.report.score_distributions), 16)
        for row in self.report.score_distributions:
            for group in ("known_correct", "known_incorrect", "unknown"):
                d = row[group]
                if d["count"] == 0:
                    self.assertIsNone(d["statistics"])
                else:
                    self.assertEqual(d["count"], d["statistics"]["count"])
        self.assertEqual(self.report.v1_baseline, self.baseline)

    def test_deterministic_ranking_and_serialization(self):
        ranking = tuple(r["identifier"] for r in sorted(self.report.candidates, key=rank_key))
        self.assertEqual(ranking, self.report.ranking)
        self.assertEqual(self.report.best_v2_development_candidate, ranking[0])
        self.assertEqual(v2_report_to_json(self.report), v2_report_to_json(self.report))

    def test_threshold_calibration_never_receives_held_unknown_identity(self):
        people = tuple(sorted(self.source.specification.unknown_ids))
        unknown_codes = np.array([100. + people.index(s.person_id) for s in self.data.unknown_samples])
        calls = []

        class Runner:
            summarize_identity = staticmethod(identity_summary)

            def infer_fold(inner, X, y, K, ky, U, n):
                def frames(predicted, scores):
                    return FrameInference(predicted, np.eye(n)[predicted],
                                          tuple((s, scores.copy()) for s in ("svm_top1", "svm_margin", "mean_5nn", "centroid_distance")))
                return frames(ky, ky.astype(float) + 2), frames(np.zeros(len(U), np.int32), unknown_codes)

        def calibrate(known, correct, unknown, kg, ug):
            expected_missing = 100. + len(calls) % 3
            self.assertEqual(set(unknown.tolist()), {100., 101., 102.} - {expected_missing})
            calls.append(True)
            return real_calibrate(known, correct, unknown, kg, ug)

        with patch("face_recognition.application.evaluate_v2_open_set_diagnostics.calibrate", calibrate):
            first = V2DiagnosticEvaluator().evaluate(self.data, self.baseline, Runner())
        self.assertEqual(len(calls), 42 * 3 * 3)
        second = V2DiagnosticEvaluator().evaluate(self.data, self.baseline, Runner())
        self.assertEqual(v2_report_to_json(first), v2_report_to_json(second))

    def test_serialized_privacy_and_no_v1_mutation(self):
        payload = v2_report_to_json(self.report)
        document = json.loads(payload)
        for person in self.source.specification.unknown_ids:
            self.assertNotIn('"' + person + '"', payload)
        for forbidden in ('"features"', "support_vectors", "centroid_vectors", "individual_scores", str(self.directory), "C:\\\\Users"):
            self.assertFalse(forbidden in payload, f"Forbidden report content: {forbidden}")
        self.assertFalse(document["v2_frozen"])
        self.assertFalse(document["production_model_persisted"])
        self.assertTrue(document["v1_immutable"])
        for path, before in self.before.items():
            self.assertEqual((sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns), before)

    def test_report_no_overwrite_incomplete_or_nan(self):
        path = self.directory / "report.json"
        write_v2_report(self.report, path, (self.processed, self.reference))
        with self.assertRaises(BaselineReportError):
            write_v2_report(self.report, path, (self.processed, self.reference))
        with self.assertRaises(BaselineReportError):
            v2_report_to_json(replace(self.report, v2_frozen=True))
        with self.assertRaises(BaselineReportError):
            v2_report_to_json(replace(self.report, candidates=()))

    def test_closed_set_metrics_known_synthetic_counts(self):
        result = identity_summary(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), ("a", "b"))
        self.assertEqual(result.confusion_matrix, ((1, 1), (0, 2)))
        self.assertEqual(result.metrics.accuracy, .75)
        self.assertAlmostEqual(result.metrics.macro_f1, (2 / 3 + .8) / 2)
        self.assertEqual(result.per_class[0].support, 2)

    def test_window_boundaries_median_strict_majority_and_tie(self):
        samples = self.data.known_samples
        n = len(samples)
        predictions = np.arange(n, dtype=np.int32) % 3
        decisions = np.zeros((n, 3))
        frames = FrameInference(predictions, decisions, (("score", np.arange(n, dtype=float)),))
        for size in (1, 3, 5, 7):
            plan, obs, agreement = _windows(samples, frames, size)
            used = []
            for w, median in zip(plan.windows, dict(obs.confidence)["score"]):
                members = [samples[i] for i in w.indices]
                self.assertEqual(len({(s.person_id, s.session_id, s.condition) for s in members}), 1)
                self.assertEqual(list(w.indices), sorted(w.indices))
                self.assertEqual(median, float(np.median(w.indices)))
                used.extend(w.indices)
            self.assertEqual(len(used), len(set(used)))
            if size == 3:
                self.assertFalse(bool(np.any(agreement * size >= 2)))
        self.assertEqual(vote_identity(np.array([0, 1]), np.array([[0., 2., 0.], [1., 0., 0.]])), (1, "tie"))
        self.assertEqual(vote_identity(np.array([0, 1]), np.zeros((2, 3))), (0, "tie"))

    def test_score_top1_and_fit_centroid_values(self):
        self.assertTrue(bool(np.array_equal(decision_top1(np.array([[1., 4., 2.]])), [4.])))
        centroid = PredictedClassCentroidScorer().fit(np.array([[0.], [2.], [10.], [12.]]), np.array([0, 0, 1, 1]), 2)
        self.assertTrue(bool(np.allclose(centroid.score(np.array([[3.], [12.]]), np.array([0, 1])), [-2., -1.])))

    def test_no_complete_window_rates_are_unavailable_not_zero(self):
        row = _known_diagnostic(np.zeros(3, np.int64))
        self.assertEqual(row["window_observations"], 0)
        self.assertIsNone(row["known_rejection_rate"])
        self.assertIsNone(row["known_correct_identification_rate"])

    def test_distance_score_and_finite_matrix_contract(self):
        scorer = KnownDistanceScorer(neighbor_counts=(5,)).fit(np.arange(6.).reshape(-1, 1), np.array([0, 0, 1, 1, 2, 2]), 3)
        self.assertTrue(bool(np.isclose(scorer.score(np.array([[10.]])).confidence[0][1][0], -7.)))
        bad = self.data.known_X.copy()
        bad[0, 0] = np.nan
        with self.assertRaises(V2StudyError):
            replace(self.data, known_X=bad)
        with self.assertRaises(V2StudyError):
            replace(self.data, known_samples=self.data.known_samples[::-1])

    def test_original_two_plus_one_session_roles_required(self):
        spec = replace(self.source.specification, train_sessions=("visit_a",), known_test_sessions=("visit_b", "reserved_visit"))
        with self.assertRaises(V2StudyError):
            study_folds(replace(self.source, specification=spec))


class V2SourceAndCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="synthetic_v2_cli_")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.processed, self.source, self.partition, self.source_path, self.partition_path, self.reference = study_fixture(self.directory)

    def args(self, report):
        return ["--processed-root", str(self.processed), "--identity-split", str(self.source_path),
                "--unknown-split", str(self.partition_path), "--v1-final-report", str(self.reference), "--report", str(report)]

    def test_changed_v1_report_and_split_binding_rejected(self):
        original = json.loads(self.reference.read_text())
        for field, value in (("status", "incomplete"), ("source_identity_split_sha256", "0" * 64)):
            bad = dict(original, **{field: value})
            self.reference.write_text(json.dumps(bad))
            with self.assertRaises(V2StudyError):
                read_v1_reference(self.reference, self.source, self.partition)
        original["operational_metrics"]["known_rejection_rate"] = float("nan")
        self.reference.write_text(json.dumps(original))
        with self.assertRaises(V2StudyError):
            read_v1_reference(self.reference, self.source, self.partition)

    def test_future_session_rejected_before_any_payload_read(self):
        baseline = read_v1_reference(self.reference, self.source, self.partition)
        loader = V2StudyLoader(self.processed)
        extra = loader.root / "known_a" / "future_visit"
        extra.mkdir()
        (extra / "center.jsonl").write_text("NOT A BIOMETRIC RECORD")
        with patch.object(loader, "_read_references", side_effect=AssertionError("Must reject catalog first")) as payload:
            with self.assertRaises(ValueError):
                loader.load_study(self.source, self.partition, baseline)
            payload.assert_not_called()

    def test_cli_success_aggregate_and_overwrite_refusal(self):
        path = self.directory / "cli_report.json"
        before = self.reference.read_bytes()
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            self.assertEqual(main(self.args(path)), 0)
        self.assertIn("V2 IS NOT FROZEN", output.getvalue())
        self.assertIn("Three known-session fits", output.getvalue())
        self.assertEqual(errors.getvalue(), "")
        self.assertEqual(before, self.reference.read_bytes())
        for person in self.source.specification.unknown_ids:
            self.assertNotIn(person, output.getvalue())
        with patch.object(V2StudyLoader, "load_study", side_effect=AssertionError("No data on overwrite")) as load:
            with redirect_stderr(io.StringIO()):
                self.assertEqual(main(self.args(path)), 1)
            load.assert_not_called()

    def test_cli_invalid_reference_sanitized_and_no_payload_open(self):
        self.reference.write_text("MALFORMED secret-sentinel")
        errors = io.StringIO()
        with patch.object(V2StudyLoader, "load_study") as load, redirect_stderr(errors):
            self.assertEqual(main(self.args(self.directory / "bad.json")), 1)
            load.assert_not_called()
        self.assertNotIn("secret-sentinel", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
