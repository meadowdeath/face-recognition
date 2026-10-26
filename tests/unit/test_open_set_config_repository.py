from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from face_recognition.application.freeze_open_set_config import freeze_development_configuration
from face_recognition.config.settings import PROJECT_ROOT
from face_recognition.domain.models.open_set_config import ConfigurationFreezeError
from face_recognition.infrastructure.persistence.baseline_evaluation_report import BaselineReportError
from face_recognition.infrastructure.persistence.open_set_config_repository import (
    frozen_configuration_to_json, read_aggregate_development_report, write_frozen_configuration,
)
from tests.open_set_config_fakes import synthetic_development_report, synthetic_report_bytes


class OpenSetConfigRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="synthetic_configuration_")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.source = self.directory / "development.json"
        self.source.write_bytes(synthetic_report_bytes())
        self.output = self.directory / "models/open_set_config_v1.json"

    def configuration(self):
        document = read_aggregate_development_report(self.source)
        return freeze_development_configuration(document.payload, document.sha256)

    def test_exact_byte_digest_deterministic_and_modified_source_changes_digest(self):
        content = self.source.read_bytes()
        first, second = read_aggregate_development_report(self.source), read_aggregate_development_report(self.source)
        self.assertEqual(first.sha256, sha256(content).hexdigest())
        self.assertEqual(first.sha256, second.sha256)
        config = self.configuration()
        self.assertEqual(config.provenance.sha256, first.sha256)
        self.source.write_bytes(content + b"\n")
        changed = read_aggregate_development_report(self.source)
        self.assertNotEqual(changed.sha256, first.sha256)
        self.assertEqual(changed.payload, first.payload)

    def test_serialization_and_writing_do_not_modify_source_and_are_deterministic(self):
        before = self.source.read_bytes(), self.source.stat().st_mtime_ns
        config = self.configuration()
        write_frozen_configuration(config, self.output, self.source)
        self.assertEqual(self.output.read_text(encoding="utf-8"), frozen_configuration_to_json(config))
        self.assertEqual((self.source.read_bytes(), self.source.stat().st_mtime_ns), before)
        data = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertIsNone(data["final_numeric_threshold"])
        self.assertEqual(data["status"], "development_configuration_frozen")
        self.assertEqual(data["completion_state"], "frozen")
        self.assertFalse(data["final_holdouts_evaluated"])
        self.assertEqual(data["protected_partitions"], ["known_test", "unknown_holdout"])
        self.assertEqual(data["architecture"]["svm_parameters"]["gamma"], .001)
        self.assertEqual(data["architecture"]["scaler_parameters"]["with_std"], True)

    def test_no_ids_paths_vectors_fitted_values_or_numeric_fold_thresholds_copied(self):
        report = synthetic_development_report()
        report["extra_private_metadata"] = {"real_name": "SYNTHETIC_PRIVATE_MARKER", "path": str(self.directory)}
        self.source.write_text(json.dumps(report), encoding="utf-8")
        text = frozen_configuration_to_json(self.configuration())
        data = json.loads(text)
        for forbidden in ("synthetic_known", "synthetic_cal", "synthetic_hold", "SYNTHETIC_PRIVATE_MARKER", str(self.directory)):
            self.assertNotIn(forbidden, text)

        def inspect(value):
            if isinstance(value, dict):
                self.assertFalse(set(value) & {"features", "landmarks", "normalized_landmarks", "selected_feature_values", "anova_scores",
                                               "scaler_statistics", "support_vectors", "support_vectors_", "reference_matrix", "real_name", "label_order"})
                for key, child in value.items():
                    if key.endswith("threshold"):
                        self.assertFalse(isinstance(child, (int, float)))
                    inspect(child)
            elif isinstance(value, list):
                for child in value:
                    inspect(child)
            elif isinstance(value, str):
                self.assertFalse(Path(value).is_absolute())
        inspect(data)

    def test_completed_output_refused_and_unsafe_targets_refused(self):
        config = self.configuration()
        write_frozen_configuration(config, self.output, self.source)
        before = self.output.read_bytes()
        with self.assertRaisesRegex(BaselineReportError, "already exists"):
            write_frozen_configuration(config, self.output, self.source)
        self.assertTrue(before == self.output.read_bytes())
        for target in (self.source, PROJECT_ROOT / "models/open_set_config_v1.json"):
            with self.assertRaises(BaselineReportError):
                write_frozen_configuration(config, target, self.source)

    def test_malformed_duplicate_nonfinite_json_and_jsonl_are_refused(self):
        for content in (b"not JSON", b"[]", b'{"schema_version":1,"schema_version":2}', b'{"value":NaN}', b'{"value":Infinity}'):
            self.source.write_bytes(content)
            with self.assertRaises(ConfigurationFreezeError):
                read_aggregate_development_report(self.source)
        with patch.object(Path, "read_bytes") as read:
            with self.assertRaises(ConfigurationFreezeError):
                read_aggregate_development_report(self.directory / "not_a_report.jsonl")
            read.assert_not_called()

    def test_atomic_publication_failure_leaves_no_partial_config(self):
        config = self.configuration()
        with patch("face_recognition.infrastructure.persistence.baseline_evaluation_report.os.link", side_effect=OSError("private")):
            with self.assertRaises(BaselineReportError):
                write_frozen_configuration(config, self.output, self.source)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.output.parent.iterdir()), [])
