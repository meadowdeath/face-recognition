from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from hashlib import sha256
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from face_recognition.application.evaluate_final_v2_open_set import FinalV2OpenSetEvaluator
from face_recognition.application.final_evaluation_data import FinalEvaluationError
from face_recognition.domain.models.final_v2_evaluation import FinalV2Context, FinalV2Session
from face_recognition.infrastructure.classifiers.sklearn_final_open_set import FinalInference
from face_recognition.infrastructure.classifiers.sklearn_final_v2_open_set import SklearnFinalV2OpenSetRunner
from face_recognition.infrastructure.classifiers.sklearn_temporal_open_set import FrameInference
from face_recognition.infrastructure.classifiers.sklearn_v2_diagnostics import identity_summary
from face_recognition.infrastructure.persistence.final_v2_evaluation_report import final_v2_report_to_json, write_final_v2_report
from face_recognition.infrastructure.persistence.final_v2_holdout_repository import FinalV2HoldoutLoader, read_final_v2_manifest, read_v2_final_inputs, write_final_v2_manifest
from face_recognition.infrastructure.persistence.v2_model_repository import load_v2_open_set_model, v2_model_fingerprints
from face_recognition.presentation.cli.create_final_v2_holdout import main as create_main
from face_recognition.presentation.cli.evaluate_final_v2_open_set import main as evaluate_main
from tests.final_v2_fakes import final_v2_fixture


class FinalV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory(prefix="synthetic_final_v2_")
        cls.directory = Path(cls.temporary.name)
        cls.paths = final_v2_fixture(cls.directory)
        cls.root, cls.source, cls.unknown, cls.model_dir, cls.config, cls.development = cls.paths
        cls.inputs = read_v2_final_inputs(cls.source, cls.unknown, cls.config, cls.development, cls.model_dir)
        cls.loader = FinalV2HoldoutLoader(cls.root)
        cls.manifest = cls.loader.catalog_manifest(cls.inputs)
        cls.manifest_path = cls.directory / "final_holdout.json"
        write_final_v2_manifest(cls.manifest, cls.manifest_path, cls.paths)
        cls.manifest_digest = sha256(cls.manifest_path.read_bytes()).hexdigest()
        cls.model = load_v2_open_set_model(cls.model_dir, cls.inputs.config, cls.inputs.config_sha256)
        cls.context = FinalV2Context(cls.inputs.config, cls.inputs.config_sha256, cls.inputs.config.provenance.sha256,
                    cls.inputs.config.provenance.original_identity_split_sha256, cls.inputs.unknown_split_sha256,
                    cls.manifest_digest, cls.manifest.processing_manifest_sha256, cls.inputs.model_fingerprints,
                    cls.model.threshold, cls.model.manifest["calibration_analysis"]["roc_auc"],
                    cls.model.manifest["calibration_analysis"]["calibration"]["metrics"]["binary_balanced_accuracy"])
        cls.data = cls.loader.load_final(cls.manifest, cls.inputs)
        cls.result = FinalV2OpenSetEvaluator(SklearnFinalV2OpenSetRunner(cls.model)).evaluate(cls.data, cls.context)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def args(self, report=None):
        args = ["--processed-root", str(self.root), "--identity-split", str(self.source), "--unknown-split", str(self.unknown),
                "--model-dir", str(self.model_dir), "--frozen-config", str(self.config), "--development-report", str(self.development)]
        if report is not None:
            return args + ["--final-holdout", str(self.manifest_path), "--report", str(report)]
        return args + ["--output", str(self.directory / "catalog_only.json")]

    def test_valid_independent_counts_metrics_and_frozen_policy(self):
        r = self.result
        self.assertEqual((r.known_final.identities, r.known_final.sessions, r.known_final.samples), (2, 2, 32))
        self.assertEqual((r.unknown_final.identities, r.unknown_final.sessions, r.unknown_final.samples), (2, 3, 48))
        self.assertEqual((r.known_windows.windows, r.unknown_windows.windows), (4, 6))
        self.assertEqual((r.known_windows.discarded_samples, r.unknown_windows.discarded_samples), (4, 6))
        self.assertEqual(r.closed_set_known_frames.samples, 32)
        self.assertEqual(r.closed_set_known_frames.label_order, self.model.parts.class_ids)
        self.assertAlmostEqual(r.operational_metrics.known_acceptance_rate + r.operational_metrics.known_rejection_rate, 1.)
        self.assertAlmostEqual(r.operational_metrics.unknown_rejection_rate + r.operational_metrics.unknown_false_acceptance_rate, 1.)
        self.assertEqual(sum(sum(row) for row in r.closed_set_known_frames.confusion_matrix), 32)
        self.assertEqual(len(r.conditions), 2)
        self.assertEqual(sum(c.known_source_samples for c in r.conditions), 32)
        self.assertEqual(r.context.threshold, self.model.threshold)
        self.assertAlmostEqual(r.delta_final_minus_development.roc_auc, r.final_holdout.roc_auc - r.development_candidate.roc_auc)

    def test_builder_reads_no_jsonl_and_never_deserializes_estimators(self):
        opened = []
        original = Path.open
        def open_file(path, *args, **kwargs):
            opened.append(path)
            return original(path, *args, **kwargs)
        output = io.StringIO()
        with patch.object(Path, "open", open_file), patch("joblib.load", side_effect=AssertionError("No unpickle")), redirect_stdout(output):
            self.assertEqual(create_main(self.args()), 0)
            self.assertEqual(create_main(self.args() + ["--validate-only"]), 0)
        self.assertFalse(any(p.suffix == ".jsonl" for p in opened))
        self.assertIn("no JSONL values inspected", output.getvalue())

    def test_development_session_missing_identity_extra_known_and_role_overlap_fail_before_payload(self):
        known = self.manifest.known_sessions
        mutations = (
            replace(self.manifest, known_sessions=(FinalV2Session(known[0].person_id, "visit_a"),) + known[1:]),
            replace(self.manifest, known_sessions=known[1:]),
            replace(self.manifest, known_sessions=tuple(sorted(known + (FinalV2Session("unexpected_known", "independent_visit"),)))),
            replace(self.manifest, unknown_sessions=tuple(sorted(self.manifest.unknown_sessions + (known[0],)))),
            replace(self.manifest, unknown_sessions=(FinalV2Session("old_unknown_a", "brand_new_visit"),)),
        )
        for manifest in mutations:
            with patch.object(self.loader, "_read_references", side_effect=AssertionError("No payload")) as read:
                with self.assertRaises(FinalEvaluationError):
                    self.loader.load_final(manifest, self.inputs)
                read.assert_not_called()

    def test_no_fit_or_operational_calibration_in_final_cli(self):
        report = self.directory / "no_refit_report.json"
        with patch.object(SelectKBest, "fit", side_effect=AssertionError("No selector fit")), \
                patch.object(StandardScaler, "fit", side_effect=AssertionError("No scaler fit")), \
                patch.object(SVC, "fit", side_effect=AssertionError("No SVM fit")), \
                patch.object(Pipeline, "fit", side_effect=AssertionError("No pipeline fit")), \
                patch("face_recognition.application.rejection_metrics.calibrate_threshold", side_effect=AssertionError("No calibration")), \
                patch("face_recognition.application.rejection_score_analysis.analyze_scores", side_effect=AssertionError("No search")), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(evaluate_main(self.args(report)), 0)
        payload = json.loads(report.read_text())
        self.assertEqual(payload["context"]["threshold"], self.model.threshold)
        self.assertFalse(payload["fitting_performed"] or payload["threshold_calibration_performed"])

    def test_eer_cutpoint_does_not_replace_persisted_threshold(self):
        data = self.data
        class Runner:
            class_ids = data.class_ids
            frozen_threshold = .95
            versions = self.result.evaluation_versions
            def infer(inner, K, y, U):
                k = FrameInference(y, np.eye(len(data.class_ids))[y] * .9, (("svm_top1", np.full(len(y), .9)),))
                u = FrameInference(np.zeros(len(U), np.int32), np.tile([.8, 0.], (len(U), 1)), (("svm_top1", np.full(len(U), .8)),))
                return FinalInference(k, u, identity_summary(y, y, data.class_ids))
        result = FinalV2OpenSetEvaluator(Runner()).evaluate(data, replace(self.context, threshold=.95))
        self.assertEqual(result.operational_metrics.known_acceptance_rate, 0.)
        self.assertEqual(result.operational_metrics.known_rejection_rate, 1.)
        self.assertEqual(result.operational_metrics.unknown_rejection_rate, 1.)
        self.assertEqual(result.operational_metrics.binary_balanced_accuracy, .5)
        self.assertNotEqual(result.temporal_score_diagnostics.equal_error_point.threshold, .95)

    def test_model_and_input_artifacts_unchanged(self):
        before = {p: (sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns) for p in self.model_dir.iterdir()}
        FinalV2OpenSetEvaluator(SklearnFinalV2OpenSetRunner(self.model)).evaluate(self.data, self.context)
        self.assertEqual(v2_model_fingerprints(self.model_dir), self.context.model_fingerprints)
        self.assertEqual(before, {p: (sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns) for p in self.model_dir.iterdir()})

    def test_report_atomic_overwrite_refusal_and_privacy(self):
        path = self.directory / "final_report.json"
        write_final_v2_report(self.result, path, self.paths + (self.manifest_path,), self.model_dir)
        before = path.read_bytes()
        with self.assertRaises(ValueError):
            write_final_v2_report(self.result, path, self.paths, self.model_dir)
        self.assertEqual(before, path.read_bytes())
        text = before.decode()
        for forbidden in ('"features"', '"landmarks"', '"per_sample_scores"', '"predictions"', str(self.directory), "new_unknown_a", "new_unknown_b"):
            self.assertFalse(forbidden in text, f"Unexpected report content: {forbidden}")
        self.assertIn("Independent final V2 generalization", text)
        self.assertEqual(final_v2_report_to_json(self.result), final_v2_report_to_json(self.result))
        with self.assertRaises(FinalEvaluationError):
            final_v2_report_to_json(replace(self.result, threshold_calibration_performed=True))
        with self.assertRaises(FinalEvaluationError):
            final_v2_report_to_json(replace(self.result, experiment="not_a_final_v2_report"))
        with self.assertRaises(FinalEvaluationError):
            write_final_v2_report(replace(self.result, context=replace(self.context, threshold=self.context.threshold + 1)),
                                  self.directory / "wrong_threshold_report.json", self.paths, self.model_dir)

    def test_existing_report_refused_before_final_loader(self):
        path = self.directory / "exists.json"
        path.write_text("EXISTING COMPLETED REPORT")
        with patch.object(FinalV2HoldoutLoader, "load_final", side_effect=AssertionError("No payload")) as read, redirect_stderr(io.StringIO()):
            self.assertEqual(evaluate_main(self.args(path)), 1)
            read.assert_not_called()

    def test_changed_model_after_evaluation_refuses_publication(self):
        original = FinalV2OpenSetEvaluator.evaluate
        def evaluate(evaluator, data, context):
            result = original(evaluator, data, context)
            # Keep the replacement active through the CLI's post-evaluation check.
            self.changed_patch = patch("face_recognition.presentation.cli.evaluate_final_v2_open_set.v2_model_fingerprints", return_value=(("changed", "c" * 64),))
            self.changed_patch.start()
            self.addCleanup(self.changed_patch.stop)
            return result
        path = self.directory / "changed_report.json"
        with patch.object(FinalV2OpenSetEvaluator, "evaluate", evaluate), redirect_stderr(io.StringIO()):
            self.assertEqual(evaluate_main(self.args(path)), 1)
        self.changed_patch.stop()
        self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
