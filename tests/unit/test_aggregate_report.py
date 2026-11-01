import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from face_recognition.config.settings import PROJECT_ROOT
from face_recognition.infrastructure.persistence.aggregate_report import (
    AggregateReportError, validate_aggregate_report_target, write_aggregate_report_json,
)


class AggregateReportTests(unittest.TestCase):
    def test_external_atomic_publication_and_overwrite_refusal(self):
        with TemporaryDirectory(prefix="synthetic_aggregate_") as folder:
            root = Path(folder)
            inputs, report = root / "inputs", root / "reports" / "summary.json"
            inputs.mkdir()
            write_aggregate_report_json('{"count": 3}\n', report, (inputs,))
            self.assertEqual(json.loads(report.read_text(encoding="utf-8")), {"count": 3})
            before = report.read_bytes()
            with self.assertRaises(AggregateReportError):
                write_aggregate_report_json('{"count": 4}\n', report, (inputs,))
            self.assertEqual(report.read_bytes(), before)
            self.assertEqual(tuple(report.parent.iterdir()), (report,))

    def test_repository_and_overlapping_targets_rejected(self):
        with self.assertRaises(AggregateReportError):
            validate_aggregate_report_target(PROJECT_ROOT / "synthetic_summary.json", ())
        with TemporaryDirectory(prefix="synthetic_aggregate_") as folder:
            root = Path(folder)
            for report, inputs in ((root / "input" / "report.json", (root / "input",)),
                                   (root / "report", (root / "report" / "input",))):
                with self.assertRaises(AggregateReportError):
                    validate_aggregate_report_target(report, inputs)

    def test_failed_publication_leaves_no_completed_or_staging_report(self):
        with TemporaryDirectory(prefix="synthetic_aggregate_") as folder:
            root = Path(folder)
            report = root / "summary.json"
            with patch("face_recognition.infrastructure.persistence.aggregate_report.os.link", side_effect=OSError("synthetic failure")):
                with self.assertRaises(AggregateReportError):
                    write_aggregate_report_json("{}\n", report, ())
            self.assertFalse(report.exists())
            self.assertEqual(tuple(root.iterdir()), ())
