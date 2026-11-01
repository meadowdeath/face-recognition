from dataclasses import asdict, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from face_recognition.application.create_identity_split import IdentitySplitError, IdentitySplitPlanner
from face_recognition.domain.models.dataset import CONDITIONS
from face_recognition.domain.models.identity_split import IdentityPartition, IdentitySplitSpec
from face_recognition.infrastructure.persistence.feature_metadata_reader import FeatureMetadataReader
from tests.identity_split_fakes import MemoryFeatureMetadataSource, write_feature_dataset


class IdentitySplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory(prefix="synthetic_identity_split_")
        self.addCleanup(self.temporary.cleanup)
        self.processed = Path(self.temporary.name) / "processed"
        self.people = {"known_alpha": ("visit_a", "visit_b", "visit_c"),
                       "known_beta": ("visit_a", "visit_b", "visit_c"),
                       "unknown_gamma": ("other_a", "other_b")}
        write_feature_dataset(self.processed, self.people)
        self.spec = IdentitySplitSpec(("known_beta", "known_alpha"), ("unknown_gamma",),
                                     ("visit_b", "visit_a"), ("visit_c",))
        self.source = FeatureMetadataReader(self.processed)
        self.planner = IdentitySplitPlanner()

    def test_known_unknown_and_whole_session_assignments_have_correct_counts(self) -> None:
        manifest = self.planner.plan(self.source, self.spec)
        self.assertEqual((manifest.known_identity_count, manifest.unknown_identity_count), (2, 1))
        counts = {row.partition: row for row in manifest.partition_counts}
        self.assertEqual(counts[IdentityPartition.KNOWN_TRAIN_POOL].samples, 16)
        self.assertEqual(counts[IdentityPartition.KNOWN_TEST].samples, 8)
        self.assertEqual(counts[IdentityPartition.UNKNOWN_POOL].samples, 8)
        self.assertEqual(counts[IdentityPartition.KNOWN_TRAIN_POOL].sessions, 4)
        self.assertEqual(counts[IdentityPartition.KNOWN_TEST].sessions, 2)
        self.assertEqual(counts[IdentityPartition.UNKNOWN_POOL].sessions, 2)
        self.assertEqual(manifest.total_sample_count, 32)
        for sample in manifest.samples:
            if sample.person_id == "unknown_gamma":
                self.assertEqual(sample.partition, IdentityPartition.UNKNOWN_POOL)
            elif sample.session_id == "visit_c":
                self.assertEqual(sample.partition, IdentityPartition.KNOWN_TEST)
            else:
                self.assertEqual(sample.partition, IdentityPartition.KNOWN_TRAIN_POOL)

    def test_no_session_or_sample_leaks_across_partitions_and_all_samples_are_accounted_for(self) -> None:
        manifest = self.planner.plan(self.source, self.spec)
        seen, sessions = set(), {}
        for sample in manifest.samples:
            key = (sample.person_id, sample.session_id, sample.condition, sample.sample_index)
            self.assertNotIn(key, seen)
            seen.add(key)
            session_key = key[:2]
            sessions.setdefault(session_key, set()).add(sample.partition)
        self.assertTrue(all(len(partitions) == 1 for partitions in sessions.values()))
        source_keys = {(r.provenance.person_id, r.provenance.session_id, r.provenance.condition, r.provenance.sample_index)
                       for r in self.source.iter_metadata()}
        self.assertEqual(seen, source_keys)
        self.assertEqual(sum(row.samples for row in manifest.partition_counts), len(source_keys))

    def test_order_is_deterministic_regardless_of_role_order_or_input_metadata_order(self) -> None:
        records = tuple(self.source.iter_metadata())
        first = self.planner.plan(MemoryFeatureMetadataSource(records), self.spec)
        second = self.planner.plan(MemoryFeatureMetadataSource(reversed(records)),
                                   IdentitySplitSpec(tuple(reversed(self.spec.known_ids)), self.spec.unknown_ids,
                                                     tuple(reversed(self.spec.train_sessions)), self.spec.known_test_sessions))
        self.assertEqual(asdict(first), asdict(second))
        self.assertEqual([sample.sample_index for sample in first.samples[:2]], [0, 1])

    def test_duplicate_role_entries_overlap_and_unsafe_ids_are_rejected(self) -> None:
        cases = ((("a", "a"), ("b",), ("s1",), ("s2",)),
                 (("a",), ("b", "b"), ("s1",), ("s2",)),
                 (("a",), ("a",), ("s1",), ("s2",)),
                 (("../a",), ("b",), ("s1",), ("s2",)),
                 (("a",), ("b",), ("s1", "s1"), ("s2",)),
                 (("a",), ("b",), ("s1",), ("s1",)),
                 ((), (), ("s1",), ("s2",)))
        for args in cases:
            with self.subTest(case=cases.index(args)), self.assertRaises(ValueError):
                IdentitySplitSpec(*args)

    def test_missing_and_unassigned_identities_are_not_silently_skipped(self) -> None:
        for spec in (replace(self.spec, known_ids=(*self.spec.known_ids, "absent_identity")),
                     replace(self.spec, known_ids=("known_alpha",))):
            with self.assertRaises(IdentitySplitError):
                self.planner.plan(self.source, spec)

    def test_missing_configured_and_extra_known_sessions_are_rejected(self) -> None:
        with self.assertRaisesRegex(IdentitySplitError, "Missing configured sessions"):
            self.planner.plan(self.source, replace(self.spec, train_sessions=("visit_a", "absent_session")))
        with self.assertRaisesRegex(IdentitySplitError, "Unassigned known sessions"):
            self.planner.plan(self.source, replace(self.spec, train_sessions=("visit_a",)))

    def test_unknown_roles_are_explicit_not_inferred_from_session_counts(self) -> None:
        # A three-session identity may explicitly be unknown; every session stays reserved.
        spec = IdentitySplitSpec(("known_alpha",), ("known_beta", "unknown_gamma"), ("visit_a", "visit_b"), ("visit_c",))
        manifest = self.planner.plan(self.source, spec)
        unknown = [sample for sample in manifest.samples if sample.partition == IdentityPartition.UNKNOWN_POOL]
        self.assertEqual(len(unknown), 20)
        self.assertEqual({sample.person_id for sample in unknown}, {"known_beta", "unknown_gamma"})

    def test_duplicate_and_gapped_sample_keys_are_rejected_even_for_custom_sources(self) -> None:
        records = tuple(self.source.iter_metadata())
        with self.assertRaisesRegex(IdentitySplitError, "Duplicate"):
            self.planner.plan(MemoryFeatureMetadataSource((*records, records[0])), self.spec)
        provenance = replace(records[0].provenance, sample_index=3)
        changed = (replace(records[0], provenance=provenance), *records[1:])
        with self.assertRaises(IdentitySplitError):
            self.planner.plan(MemoryFeatureMetadataSource(changed), self.spec)

    def test_generic_participant_session_condition_and_sample_counts(self) -> None:
        processed = Path(self.temporary.name) / "other_processed"
        people = {"known_single": ("one", "two", "three", "four", "five"),
                  "unknown_single": ("only_unknown",)}
        write_feature_dataset(processed, people, conditions=CONDITIONS[:3], samples=5, landmarks=7)
        spec = IdentitySplitSpec(("known_single",), ("unknown_single",), ("one", "two", "three"), ("four", "five"))
        manifest = self.planner.plan(FeatureMetadataReader(processed), spec)
        self.assertEqual(manifest.total_sample_count, 90)
        self.assertEqual([count.samples for count in manifest.partition_counts], [45, 30, 15])
        self.assertEqual(len(manifest.session_assignments), 6)

    def test_manifest_contains_references_and_leakage_rules_not_biometric_values(self) -> None:
        manifest = self.planner.plan(self.source, self.spec)
        data = asdict(manifest)
        self.assertEqual(data["completion_state"], "completed")
        self.assertEqual(data["source_representation"], "all_xyz_v1")
        self.assertEqual(set(data["specification"]["known_ids"]), set(self.spec.known_ids))
        self.assertIn("not a final test set", json.dumps(data["leakage_constraints"]))
        for sample in manifest.samples:
            self.assertTrue(sample.source_relative_path.startswith("all_xyz_v1/"))
            self.assertFalse(Path(sample.source_relative_path).is_absolute())
        def check(value):
            if isinstance(value, dict):
                self.assertFalse(set(value) & {"features", "landmarks", "normalized_landmarks"})
                for child in value.values():
                    check(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    check(child)
            elif isinstance(value, str):
                self.assertNotIn(str(self.temporary.name), value)
        check(data)
