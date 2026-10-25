from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from face_recognition.application.unknown_rejection_split import UnknownRejectionError, plan_unknown_split, validate_unknown_split
from face_recognition.domain.models.identity_split import IdentityPartition
from face_recognition.domain.models.unknown_rejection import UnknownSplitSpec
from face_recognition.infrastructure.persistence.baseline_evaluation_report import BaselineReportError
from face_recognition.infrastructure.persistence.unknown_rejection_repository import read_unknown_split, unknown_split_to_json, write_unknown_split
from tests.unknown_rejection_fakes import rejection_fixture


class UnknownRejectionSplitTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="synthetic_unknown_split_")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.processed, self.source, self.partition, self.source_path, self.path = rejection_fixture(self.directory)

    def test_entire_identities_sessions_samples_disjoint_and_all_accounted_for(self):
        self.assertEqual((self.partition.calibration.identities, self.partition.calibration.sessions, self.partition.calibration.samples), (1, 2, 12))
        self.assertEqual((self.partition.holdout.identities, self.partition.holdout.sessions, self.partition.holdout.samples), (2, 3, 18))
        assigned = {}
        for row in self.source.samples:
            if row.partition is IdentityPartition.UNKNOWN_POOL:
                role = "calibration" if row.person_id in self.partition.specification.calibration_ids else "holdout"
                assigned.setdefault(row.person_id, set()).add(role)
        self.assertEqual(set(assigned), set(self.source.specification.unknown_ids))
        self.assertTrue(all(len(roles) == 1 for roles in assigned.values()))
        self.assertEqual(self.partition.calibration.samples + self.partition.holdout.samples, 30)

    def test_known_missing_overlap_duplicate_and_unsafe_id_assignments_rejected(self):
        for calibration, holdout in ((("known_a",), ("unknown_hold_a", "unknown_hold_b")),
                                     (("unknown_cal",), ("unknown_hold_a",)),
                                     (("absent",), ("unknown_cal", "unknown_hold_a", "unknown_hold_b")),
                                     (("unknown_cal",), ("unknown_cal", "unknown_hold_a", "unknown_hold_b")),
                                     (("unknown_cal", "unknown_cal"), ("unknown_hold_a", "unknown_hold_b")),
                                     (("../bad",), ("unknown_hold_a", "unknown_hold_b")),
                                     ((), ("unknown_cal", "unknown_hold_a", "unknown_hold_b"))):
            with self.assertRaises(ValueError):
                plan_unknown_split(self.source, UnknownSplitSpec(calibration, holdout))

    def test_deterministic_round_trip_and_source_metadata_binding_without_payload_reads(self):
        original = Path.open

        def metadata_only(path, *args, **kwargs):
            self.assertNotEqual(path.suffix, ".jsonl")
            return original(path, *args, **kwargs)

        with patch.object(Path, "open", metadata_only):
            decoded = read_unknown_split(self.path, self.source)
            reversed_roles = UnknownSplitSpec(self.partition.specification.calibration_ids,
                                              tuple(reversed(self.partition.specification.holdout_ids)))
            second = plan_unknown_split(self.source, reversed_roles)
        self.assertEqual(decoded, self.partition)
        self.assertEqual(unknown_split_to_json(second), unknown_split_to_json(decoded))
        data = json.loads(unknown_split_to_json(decoded))
        self.assertEqual(len(data["source_identity_split_sha256"]), 64)
        self.assertNotIn(str(self.directory), json.dumps(data))
        self.assertFalse(set(data) & {"features", "landmarks", "samples"})

    def test_completed_split_overwrite_refused(self):
        before = self.path.read_bytes()
        with self.assertRaisesRegex(BaselineReportError, "already exists"):
            write_unknown_split(self.partition, self.source, self.path, self.source_path)
        self.assertTrue(before == self.path.read_bytes())

    def test_tampered_source_hash_counts_schema_and_noncanonical_order_rejected(self):
        original = json.loads(self.path.read_text(encoding="utf-8"))
        for change in (lambda d: d.__setitem__("source_identity_split_sha256", "different"),
                       lambda d: d["calibration"].__setitem__("samples", 1),
                       lambda d: d["calibration"].__setitem__("identities", True),
                       lambda d: d.__setitem__("schema_version", 2),
                       lambda d: d["specification"]["holdout_ids"].reverse(),
                       lambda d: d.__setitem__("features", [123.456])):
            data = json.loads(json.dumps(original))
            change(data)
            self.path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(UnknownRejectionError) as caught:
                read_unknown_split(self.path, self.source)
            self.assertNotIn("123.456", str(caught.exception))
        with self.assertRaises(UnknownRejectionError):
            validate_unknown_split(self.source, replace(self.partition, source_total_samples=0))
