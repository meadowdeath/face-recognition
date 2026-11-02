from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from face_recognition.presentation.cli import analyze_v3_mahalanobis_diagnostics as cli
from tests.v3_mahalanobis_diagnostics_fakes import DiagnosticReplayRunner, diagnostic_fixture
from tests.v3_mahalanobis_fakes import RecordingSameModelRunner


class MahalanobisDiagnosticsCLITests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="synthetic_v3_diagnostic_cli_")
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.data, self.processed, self.protocol, self.source, _, _ = diagnostic_fixture(self.folder)
        self.output = self.folder / "reports" / "diagnostics.json"

    def invoke(self, target=None):
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.main(["--processed-root", str(self.processed), "--protocol", str(self.protocol),
                "--evaluation-report", str(self.source), "--report", str(target or self.output)])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_success_is_aggregate_only_deterministic_and_leaves_evidence_unchanged(self):
        inputs = (self.protocol, self.source, self.folder / "selection.json")
        before = [(p.read_bytes(), p.stat().st_mtime_ns) for p in inputs]
        with patch.object(cli, "load_v3_payloads", return_value=self.data), \
                patch.object(cli, "SklearnV3SameModelRunner", side_effect=DiagnosticReplayRunner):
            status, stdout, stderr = self.invoke()
            self.assertEqual((status, stderr), (0, ""))
            self.assertEqual(self.invoke(self.folder / "second.json")[0], 0)
        self.assertEqual(self.output.read_bytes(), (self.folder / "second.json").read_bytes())
        self.assertIn("no model selection/calibration decisions changed", stdout)
        self.assertIn("not a new evaluation", stdout)
        self.assertNotIn(str(self.folder), stdout)
        self.assertNotIn("known_00", stdout)
        self.assertNotIn("[[-", stdout)
        self.assertEqual(before, [(p.read_bytes(), p.stat().st_mtime_ns) for p in inputs])
        raw = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(raw["threshold_transport"], "none_same_model_calibration")
        self.assertEqual(raw["provenance"]["evaluation_report_reference"], self.source.name)

    def test_overwrite_or_input_alias_refused_before_payload_access(self):
        self.output.parent.mkdir()
        self.output.write_text("completed", encoding="utf-8")
        for target in (self.output, self.source, self.protocol):
            before = target.read_bytes(), target.stat().st_mtime_ns
            with patch.object(cli, "read_v3_cv_protocol") as reader, patch.object(cli, "load_v3_payloads") as loader:
                self.assertEqual(self.invoke(target)[0], 1)
                reader.assert_not_called(); loader.assert_not_called()
            self.assertEqual(before, (target.read_bytes(), target.stat().st_mtime_ns))

    def test_protocol_and_report_mismatch_rejected_before_payload_access(self):
        original = self.protocol.read_bytes()
        self.protocol.write_text("{}", encoding="utf-8")
        with patch.object(cli, "load_v3_payloads") as loader:
            self.assertEqual(self.invoke()[0], 1)
            loader.assert_not_called()
        self.protocol.write_bytes(original)
        for field, value in (("protocol_sha256", "0" * 64), ("completion_state", "failed")):
            raw = json.loads(self.source.read_text(encoding="utf-8")); raw[field] = value
            self.source.write_text(json.dumps(raw), encoding="utf-8")
            with patch.object(cli, "load_v3_payloads") as loader, patch.object(cli, "SklearnV3SameModelRunner") as runner:
                self.assertEqual(self.invoke()[0], 1)
                loader.assert_not_called(); runner.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_retained_strict_loader_rejects_invalid_synthetic_jsonl_before_fitting(self):
        # Catalog-only fixture contains invalid feature JSONL markers. Exercise
        # the real payload loader to prove the CLI does not bypass validation.
        with patch.object(cli, "SklearnV3SameModelRunner") as runner:
            self.assertEqual(self.invoke()[0], 1)
            runner.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_failed_fit_or_mismatching_outer_replay_never_publishes_partial_report(self):
        runner = DiagnosticReplayRunner()
        runner.fit = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("SYNTHETIC_PRIVATE_STATE"))
        for adapter in (runner, RecordingSameModelRunner()):
            with patch.object(cli, "load_v3_payloads", return_value=self.data), \
                    patch.object(cli, "SklearnV3SameModelRunner", return_value=adapter):
                status, stdout, stderr = self.invoke()
            self.assertEqual(status, 1)
            self.assertNotIn("SYNTHETIC_PRIVATE_STATE", stdout + stderr)
            self.assertFalse(self.output.exists())
            self.assertFalse(self.output.parent.exists())

    def test_source_report_changed_during_replay_prevents_publication(self):
        runner = DiagnosticReplayRunner()
        original = runner.fit
        def fit(*args, **kwargs):
            self.source.write_bytes(self.source.read_bytes() + b"\n")
            return original(*args, **kwargs)
        runner.fit = fit
        with patch.object(cli, "load_v3_payloads", return_value=self.data), patch.object(cli, "SklearnV3SameModelRunner", return_value=runner):
            self.assertEqual(self.invoke()[0], 1)
        self.assertFalse(self.output.exists())

    def test_atomic_publication_failure_leaves_no_report(self):
        with patch.object(cli, "load_v3_payloads", return_value=self.data), \
                patch.object(cli, "SklearnV3SameModelRunner", return_value=DiagnosticReplayRunner()), \
                patch("face_recognition.infrastructure.persistence.aggregate_report.os.link", side_effect=OSError("synthetic")):
            self.assertEqual(self.invoke()[0], 1)
        self.assertFalse(self.output.exists())
        self.assertEqual(tuple(self.output.parent.iterdir()), ())
