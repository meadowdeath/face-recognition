from collections import Counter
from dataclasses import replace
from itertools import permutations
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from face_recognition.application.create_v3_cv_protocol import create_v3_cv_protocol, validate_v3_cv_protocol
from face_recognition.domain.models.v3_cv_protocol import V3ProtocolError
from face_recognition.infrastructure.persistence.v3_cv_protocol_repository import read_v3_catalog, v3_protocol_to_json
from tests.v3_protocol_fakes import v3_catalog_fixture


class V3ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory(prefix="synthetic_v3_protocol_")
        cls.processed, cls.spec_path, cls.spec = v3_catalog_fixture(Path(cls.temporary.name))
        cls.catalog = read_v3_catalog(cls.processed)
        cls.protocol = create_v3_cv_protocol(cls.spec, cls.catalog)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_current_population_and_exactly_four_deterministic_outer_tests(self):
        p = self.protocol
        self.assertEqual((len(p.specification.known_ids), len(p.specification.unknown_ids)), (10, 8))
        self.assertEqual(len(p.catalog.observed_sessions), 18)
        self.assertEqual(len(p.catalog.condition_files), (10 * 4 + 8 * 2) * 13)
        self.assertEqual(tuple(f.known_test_session for f in p.outer_folds), self.spec.known_sessions)
        self.assertEqual(tuple(f.name for f in p.outer_folds), ("outer_01", "outer_02", "outer_03", "outer_04"))
        self.assertEqual(p, create_v3_cv_protocol(self.spec, self.catalog))
        self.assertEqual(v3_protocol_to_json(p), v3_protocol_to_json(create_v3_cv_protocol(self.spec, self.catalog)))

    def test_each_known_identity_session_tested_once_without_development_overlap(self):
        tested = Counter()
        all_known = {(person, session) for person in self.spec.known_ids for session in self.spec.known_sessions}
        for fold in self.protocol.outer_folds:
            dev = {(person, session) for person in self.spec.known_ids for session in fold.known_development_sessions}
            test = {(person, fold.known_test_session) for person in self.spec.known_ids}
            self.assertFalse(dev & test)
            self.assertEqual(dev | test, all_known)
            self.assertEqual({p for p, _ in test}, set(self.spec.known_ids))
            tested.update(test)
        self.assertEqual(tested, Counter({pair: 1 for pair in all_known}))

    def test_unknown_identity_groups_test_once_with_both_sessions_together(self):
        tested = Counter()
        available = {row.person_id: row.session_ids for row in self.catalog.observed_sessions}
        for fold in self.protocol.outer_folds:
            self.assertEqual(len(fold.unknown_test_ids), 2)
            self.assertEqual(len(fold.unknown_development_ids), 6)
            self.assertFalse(set(fold.unknown_test_ids) & set(fold.unknown_development_ids))
            self.assertEqual(set(fold.unknown_test_ids) | set(fold.unknown_development_ids), set(self.spec.unknown_ids))
            tested.update(fold.unknown_test_ids)
            for person in fold.unknown_test_ids:
                self.assertEqual(available[person], self.spec.unknown_sessions)
        self.assertEqual(tested, Counter({person: 1 for person in self.spec.unknown_ids}))

    def test_all_inner_fit_calibration_validation_roles_are_separate_and_inside_outer_development(self):
        for outer in self.protocol.outer_folds:
            self.assertEqual(len(outer.inner_splits), 18)
            known_roles = Counter()
            for inner in outer.inner_splits:
                fit = set(inner.known_fit_sessions)
                calibration = {inner.known_calibration_session}
                validation = {inner.known_validation_session}
                self.assertFalse(fit & calibration or fit & validation or calibration & validation)
                self.assertEqual(fit | calibration | validation, set(outer.known_development_sessions))
                self.assertNotIn(outer.known_test_session, fit | calibration | validation)
                known_roles.update([(*inner.known_fit_sessions, inner.known_calibration_session, inner.known_validation_session)])
                uc, uv = set(inner.unknown_calibration_ids), set(inner.unknown_validation_ids)
                self.assertEqual((len(uc), len(uv)), (4, 2))
                self.assertFalse(uc & uv)
                self.assertEqual(uc | uv, set(outer.unknown_development_ids))
                self.assertFalse((uc | uv) & set(outer.unknown_test_ids))
            self.assertEqual(known_roles, Counter({roles: 3 for roles in permutations(outer.known_development_sessions)}))

    def test_input_participant_session_order_does_not_change_protocol(self):
        reversed_spec = replace(self.spec, known_ids=self.spec.known_ids[::-1], unknown_ids=self.spec.unknown_ids[::-1],
                                known_sessions=self.spec.known_sessions[::-1], unknown_sessions=self.spec.unknown_sessions[::-1])
        self.assertEqual(self.protocol, create_v3_cv_protocol(reversed_spec, self.catalog))

    def test_role_overlap_duplicate_assignments_and_incompatible_unknown_counts_rejected(self):
        changes = ({"unknown_ids": (self.spec.known_ids[0], *self.spec.unknown_ids[1:])},
                   {"known_ids": (*self.spec.known_ids, self.spec.known_ids[0])},
                   {"unknown_ids": (*self.spec.unknown_ids[:-1], self.spec.unknown_ids[0])},
                   {"unknown_ids": self.spec.unknown_ids[:-1]}, {"unknown_ids": (*self.spec.unknown_ids, "extra")})
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(self.spec, **change)

    def test_missing_extra_and_wrong_configured_sessions_rejected(self):
        for person in (self.spec.known_ids[0], self.spec.unknown_ids[0]):
            row = next(row for row in self.catalog.observed_sessions if row.person_id == person)
            for sessions in (row.session_ids[:-1], (*row.session_ids, "extra"), (*row.session_ids[:-1], "different")):
                observed = tuple(replace(r, session_ids=sessions) if r.person_id == person else r for r in self.catalog.observed_sessions)
                with self.subTest(person=person, sessions=sessions), self.assertRaises(V3ProtocolError):
                    create_v3_cv_protocol(self.spec, replace(self.catalog, observed_sessions=observed))
        for change in ({"known_sessions": self.spec.known_sessions[:-1]}, {"unknown_sessions": (*self.spec.unknown_sessions, "extra")}):
            with self.assertRaises(V3ProtocolError):
                replace(self.spec, **change)

    def test_missing_condition_empty_file_extra_identity_and_unsafe_reference_rejected(self):
        row = self.catalog.condition_files[0]
        variants = (replace(self.catalog, condition_files=self.catalog.condition_files[1:]),
                    replace(self.catalog, condition_files=(replace(row, size_bytes=0), *self.catalog.condition_files[1:])),
                    replace(self.catalog, condition_files=(replace(row, source_relative_path="../unsafe.jsonl"), *self.catalog.condition_files[1:])),
                    replace(self.catalog, observed_sessions=self.catalog.observed_sessions[:-1]))
        for catalog in variants:
            with self.assertRaises(V3ProtocolError):
                create_v3_cv_protocol(self.spec, catalog)

    def test_illegal_inner_split_fails_protocol_regeneration(self):
        outer = self.protocol.outer_folds[0]
        inner = replace(outer.inner_splits[0], known_fit_sessions=(outer.inner_splits[0].known_validation_session,))
        changed = replace(self.protocol, outer_folds=(replace(outer, inner_splits=(inner, *outer.inner_splits[1:])), *self.protocol.outer_folds[1:]))
        with self.assertRaises(V3ProtocolError):
            validate_v3_cv_protocol(changed)
