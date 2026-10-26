from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from sklearn.pipeline import Pipeline

from face_recognition.application.freeze_v2_open_set_config import freeze_v2_configuration
from face_recognition.config.settings import PROJECT_ROOT
from face_recognition.domain.models.open_set_config import ConfigurationFreezeError
from face_recognition.infrastructure.persistence.baseline_evaluation_report import BaselineReportError
from face_recognition.infrastructure.persistence.open_set_config_repository import read_aggregate_development_report
from face_recognition.infrastructure.persistence.v2_open_set_config_repository import (
    frozen_v2_configuration_from_json, frozen_v2_configuration_to_json,
    read_frozen_v2_configuration, write_frozen_v2_configuration,
)
from face_recognition.presentation.cli.freeze_v2_open_set_config import main
from tests.v2_open_set_config_fakes import synthetic_v2_report


class V2OpenSetConfigRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="synthetic_v2_freeze_")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.source = self.directory / "diagnostics.json"
        self.output = self.directory / "models" / "open_set_config_v2.json"
        self.source.write_text(json.dumps(synthetic_v2_report()), encoding="utf-8")
        self.document = read_aggregate_development_report(self.source)
        self.config = freeze_v2_configuration(self.document.payload, self.document.sha256)

    def args(self):
        return ["--diagnostic-report", str(self.source), "--output", str(self.output)]

    def test_external_atomic_write_round_trip_and_digest(self):
        before, modified = self.source.read_bytes(), self.source.stat().st_mtime_ns
        write_frozen_v2_configuration(self.config, self.output, self.source)
        self.assertEqual(read_frozen_v2_configuration(self.output), self.config)
        self.assertEqual(self.config.provenance.sha256, sha256(before).hexdigest())
        self.assertEqual(before, self.source.read_bytes())
        self.assertEqual(modified, self.source.stat().st_mtime_ns)
        self.assertEqual(list(self.output.parent.glob("*.tmp")), [])
        self.assertEqual(self.output.read_text(), frozen_v2_configuration_to_json(self.config))

    def test_overwrite_refused_and_original_bytes_retained(self):
        write_frozen_v2_configuration(self.config, self.output, self.source)
        before = self.output.read_bytes()
        with self.assertRaises(BaselineReportError):
            write_frozen_v2_configuration(self.config, self.output, self.source)
        self.assertEqual(before, self.output.read_bytes())

    def test_repository_local_output_and_source_overlap_refused(self):
        for target in (PROJECT_ROOT / "models" / "never_write_v2.json", self.source):
            with self.assertRaises(BaselineReportError):
                write_frozen_v2_configuration(self.config, target, self.source)

    def test_exact_bytes_digest_changes_with_source_whitespace(self):
        self.source.write_text(self.source.read_text() + "\n", encoding="utf-8")
        second = read_aggregate_development_report(self.source)
        self.assertNotEqual(second.sha256, self.document.sha256)
        frozen = freeze_v2_configuration(second.payload, second.sha256)
        self.assertNotEqual(frozen.provenance.sha256, self.config.provenance.sha256)
        self.assertEqual(frozen.development_evidence, self.config.development_evidence)
        with self.assertRaises(ConfigurationFreezeError):
            write_frozen_v2_configuration(self.config, self.output, self.source)
        self.assertFalse(self.output.exists())

    def test_modified_or_incomplete_frozen_policy_rejected(self):
        original = json.loads(frozen_v2_configuration_to_json(self.config))
        for key, value in (("final_numeric_threshold", .9), ("completion_state", "incomplete"),
                            ("final_model_fitted", True), ("v2_final_evaluation_performed", True)):
            data = deepcopy(original)
            data[key] = value
            with self.assertRaises(ConfigurationFreezeError):
                frozen_v2_configuration_from_json(data)
        for key, value in (("rejection_score_family", "mean_5nn"), ("identity_consistency_gate", True), ("temporal_window_size", 5)):
            data = deepcopy(original)
            data["architecture"][key] = value
            with self.assertRaises(ConfigurationFreezeError):
                frozen_v2_configuration_from_json(data)
        data = deepcopy(original)
        del data["final_threshold_calibration_policy"]
        with self.assertRaises(ConfigurationFreezeError):
            frozen_v2_configuration_from_json(data)

    def test_publication_evidence_must_match_hashed_report(self):
        altered = replace(self.config, development_evidence=replace(self.config.development_evidence, mean_roc_auc=.91))
        with self.assertRaises(ConfigurationFreezeError):
            write_frozen_v2_configuration(altered, self.output, self.source)
        self.assertFalse(self.output.exists())

    def test_cli_aggregate_only_no_fitting_and_reads_only_source_report(self):
        output, errors, reads = io.StringIO(), io.StringIO(), []
        original_read = Path.read_bytes

        def read(path):
            reads.append(path.resolve())
            return original_read(path)

        with patch.object(Path, "read_bytes", read), patch.object(Pipeline, "fit", side_effect=AssertionError("No fitting")) as fit, \
                redirect_stdout(output), redirect_stderr(errors):
            self.assertEqual(main(self.args()), 0)
            fit.assert_not_called()
        self.assertEqual(set(reads), {self.source.resolve()})
        text = output.getvalue()
        self.assertIn("V2 OPEN-SET DEVELOPMENT CONFIGURATION FROZEN", text)
        self.assertIn("consistency gate=disabled", text)
        self.assertIn("not yet produced", text)
        self.assertIn("final evaluation performed=no", text)
        self.assertIn("biometric payloads accessed=no", text)
        self.assertEqual(errors.getvalue(), "")
        for forbidden in ("synthetic_a", "synthetic_b", "visit_a", str(self.directory), "confusion_matrix", "feature_vectors"):
            self.assertFalse(forbidden in text, f"Unexpected CLI content: {forbidden}")

    def test_cli_wrong_winner_nonzero_sanitized_and_no_output(self):
        data = self.document.payload
        data["best_v2_development_candidate"] = "PRIVATE-INVALID-CANDIDATE"
        self.source.write_text(json.dumps(data))
        errors = io.StringIO()
        with redirect_stderr(errors):
            self.assertEqual(main(self.args()), 1)
        self.assertFalse(self.output.exists())
        self.assertNotIn("PRIVATE-INVALID-CANDIDATE", errors.getvalue())

    def test_cli_overwrite_refuses_before_report_read(self):
        write_frozen_v2_configuration(self.config, self.output, self.source)
        with patch("face_recognition.presentation.cli.freeze_v2_open_set_config.read_aggregate_development_report") as read, redirect_stderr(io.StringIO()):
            self.assertEqual(main(self.args()), 1)
            read.assert_not_called()

    def test_cli_no_parameter_overrides_or_force(self):
        for extra in (("--force",), ("--k", "512"), ("--threshold", ".9")):
            with redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as result:
                    main(self.args() + list(extra))
            self.assertEqual(result.exception.code, 2)

    def test_invalid_json_nonfinite_and_duplicate_keys(self):
        for payload in ('{"x":NaN}', '{"x":1,"x":2}', 'not-json'):
            self.source.write_text(payload)
            with self.assertRaises(ConfigurationFreezeError):
                read_aggregate_development_report(self.source)
            with redirect_stderr(io.StringIO()):
                self.assertEqual(main(self.args()), 1)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
