from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest.mock import patch

from face_recognition.infrastructure.persistence.v3_cv_protocol_repository import write_v3_cv_protocol
from face_recognition.presentation.cli import evaluate_v3_novelty as cli
from tests.v3_novelty_fakes import RecordingNoveltyRunner, novelty_data


class V3NoveltyCLITests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="synthetic_v3_report_")
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.data, self.processed, self.spec_path = novelty_data(self.folder)
        self.protocol_path, self.output = self.folder / "protocol.json", self.folder / "reports" / "novelty.json"
        write_v3_cv_protocol(self.data.protocol, self.protocol_path, (self.processed, self.spec_path))

    def invoke(self):
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = cli.main(["--processed-root", str(self.processed), "--protocol", str(self.protocol_path), "--report", str(self.output)])
        return status, stdout.getvalue(), stderr.getvalue()

    def test_success_atomic_aggregate_report_and_input_immutability(self):
        before = self.protocol_path.read_bytes(), self.spec_path.read_bytes()
        with patch.object(cli, "load_v3_payloads", return_value=self.data), \
                patch.object(cli, "SklearnV3NoveltyRunner", return_value=RecordingNoveltyRunner()):
            status, stdout, stderr = self.invoke()
        self.assertEqual((status, stderr), (0, ""))
        self.assertIn("outer_04: outer evaluation complete", stdout)
        self.assertIn("INNER", stdout.upper())
        self.assertNotIn("known_00", stdout)
        self.assertNotIn(str(self.folder), stdout)
        raw = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(len(raw["outer_results"]), 4)
        self.assertEqual(tuple(name for name, _ in raw["selection_frequency"]),
                         ("svm_top1", "svm_margin", "centroid_distance", "mahalanobis_distance"))
        self.assertEqual(tuple(self.output.parent.iterdir()), (self.output,))
        self.assertEqual(before, (self.protocol_path.read_bytes(), self.spec_path.read_bytes()))
        def check(value):
            if isinstance(value, dict):
                self.assertFalse(set(value) & {"features", "landmarks", "predicted", "decisions", "keys", "_centers", "_factors", "support_vectors_"})
                for item in value.values(): check(item)
            elif isinstance(value, list):
                for item in value: check(item)
            elif isinstance(value, str):
                self.assertFalse(Path(value).is_absolute())
        check(raw)

    def test_overwrite_refused_before_protocol_or_payload_access(self):
        self.output.parent.mkdir()
        self.output.write_text("completed synthetic report", encoding="utf-8")
        before = self.output.read_bytes()
        with patch.object(cli, "read_v3_cv_protocol", side_effect=AssertionError("No protocol access")), \
                patch.object(cli, "load_v3_payloads", side_effect=AssertionError("No payload access")):
            self.assertEqual(self.invoke()[0], 1)
        self.assertEqual(self.output.read_bytes(), before)

    def test_invalid_protocol_precedes_payload_and_failed_evaluation_publishes_nothing(self):
        original = self.protocol_path.read_bytes()
        self.protocol_path.write_text("{}", encoding="utf-8")
        with patch.object(cli, "load_v3_payloads") as loader:
            self.assertEqual(self.invoke()[0], 1)
            loader.assert_not_called()
        self.protocol_path.write_bytes(original)
        failing = RecordingNoveltyRunner()
        failing.fit = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("SYNTHETIC_PRIVATE_VALUES"))
        with patch.object(cli, "load_v3_payloads", return_value=self.data), patch.object(cli, "SklearnV3NoveltyRunner", return_value=failing):
            status, stdout, stderr = self.invoke()
        self.assertEqual(status, 1)
        self.assertNotIn("SYNTHETIC_PRIVATE_VALUES", stdout + stderr)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.output.parent.exists())

    def test_changed_protocol_during_execution_refuses_publication(self):
        runner = RecordingNoveltyRunner()
        original_fit = runner.fit
        def fit(*args, **kwargs):
            self.protocol_path.write_bytes(self.protocol_path.read_bytes() + b"\n")
            return original_fit(*args, **kwargs)
        runner.fit = fit
        with patch.object(cli, "load_v3_payloads", return_value=self.data), patch.object(cli, "SklearnV3NoveltyRunner", return_value=runner):
            self.assertEqual(self.invoke()[0], 1)
        self.assertFalse(self.output.exists())
