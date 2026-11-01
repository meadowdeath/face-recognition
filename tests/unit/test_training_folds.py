from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from face_recognition.application.training_folds import (
    DevelopmentDataset, SessionFoldSpec, TrainingFoldError, TrainingRows, create_session_folds,
)
from face_recognition.domain.models.identity_split import IdentityPartition
from face_recognition.infrastructure.persistence.development_dataset_loader import DevelopmentDatasetLoader
from tests.training_fold_fakes import training_fixture


class TrainingFoldsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="synthetic_folds_")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        processed, _, manifest, _ = training_fixture(self.directory)
        self.data = DevelopmentDatasetLoader(processed).load(manifest)

    def test_fold_a_and_b_keep_whole_sessions_with_opposite_roles(self):
        first, second = create_session_folds(self.data)
        self.assertEqual((first.name, second.name), ("Fold A", "Fold B"))
        self.assertEqual((first.fit_session_ids, first.validation_session_ids), (("visit_a",), ("visit_b",)))
        self.assertEqual((second.fit_session_ids, second.validation_session_ids), (("visit_b",), ("visit_a",)))
        self.assertEqual(first.fit.indices, second.validation.indices)
        self.assertEqual(first.validation.indices, second.fit.indices)
        for fold in (first, second):
            self.assertEqual((len(fold.fit.indices), len(fold.validation.indices)), (8, 8))
            self.assertEqual((fold.fit.identity_count, fold.validation.identity_count), (2, 2))
            self.assertFalse(set(fold.fit_session_ids) & set(fold.validation_session_ids))
            self.assertFalse(set(fold.fit.metadata) & set(fold.validation.metadata))
            self.assertEqual(set(fold.fit.metadata) | set(fold.validation.metadata), set(self.data.samples))
            for rows, sessions in ((fold.fit, fold.fit_session_ids), (fold.validation, fold.validation_session_ids)):
                self.assertEqual({s.session_id for s in rows.metadata}, set(sessions))
                self.assertTrue(all(s.partition is IdentityPartition.KNOWN_TRAIN_POOL for s in rows.metadata))
                for sample, label in zip(rows.metadata, rows.y):
                    self.assertEqual(self.data.label_to_person[int(label)], sample.person_id)

    def test_opposite_fold_arrays_share_storage_without_full_matrix_copies(self):
        first, second = create_session_folds(self.data)
        for rows in (first.fit, first.validation, second.fit, second.validation):
            X, y = rows.arrays()
            self.assertTrue(np.shares_memory(X, self.data.X))
            self.assertTrue(np.shares_memory(y, self.data.y))
            self.assertFalse(X.flags.writeable)
            self.assertFalse(y.flags.writeable)
        self.assertTrue(np.shares_memory(first.fit.X, second.validation.X))

    def test_generation_is_deterministic_and_never_uses_sample_randomization(self):
        with patch("random.shuffle", side_effect=AssertionError("Random split forbidden")), \
             patch("random.sample", side_effect=AssertionError("Random split forbidden")), \
             patch("numpy.random.permutation", side_effect=AssertionError("Random split forbidden")), \
             patch("numpy.random.shuffle", side_effect=AssertionError("Random split forbidden")):
            first, second = create_session_folds(self.data), create_session_folds(self.data)
        self.assertEqual(tuple((f.name, f.fit.indices, f.validation.indices) for f in first),
                         tuple((f.name, f.fit.indices, f.validation.indices) for f in second))

    def test_immutable_wrappers_and_safe_representations(self):
        fold = create_session_folds(self.data)[0]
        for value in (fold, fold.fit, self.data):
            self.assertNotIn("array", repr(value))
            self.assertNotIn("features=([", repr(value))
        with self.assertRaises(FrozenInstanceError):
            fold.name = "changed"
        with self.assertRaises(ValueError):
            fold.fit.X[0, 0] = 0

    def test_explicit_specs_support_more_sessions_without_per_sample_splitting(self):
        processed, _, manifest, _ = training_fixture(self.directory / "three", training=("a", "b", "c"), samples=3)
        data = DevelopmentDatasetLoader(processed).load(manifest)
        with self.assertRaisesRegex(TrainingFoldError, "explicit"):
            create_session_folds(data)
        folds = create_session_folds(data, (SessionFoldSpec("Middle validation", ("c", "a"), ("b",)),))
        fold = folds[0]
        self.assertEqual(fold.fit_session_ids, ("a", "c"))
        self.assertEqual((len(fold.fit.indices), len(fold.validation.indices)), (24, 12))
        self.assertEqual({s.session_id for s in fold.fit.metadata}, {"a", "c"})
        # Noncontiguous selections gather only when arrays are requested.
        self.assertFalse(np.shares_memory(fold.fit.X, data.X))
        self.assertFalse(fold.fit.X.flags.writeable)

    def test_invalid_or_protected_session_specifications_are_rejected(self):
        cases = (("overlap", ("visit_a",), ("visit_a",)),
                 ("empty", (), ("visit_b",)),
                 ("duplicate", ("visit_a", "visit_a"), ("visit_b",)))
        for args in cases:
            with self.subTest(name=args[0]), self.assertRaises(ValueError):
                SessionFoldSpec(*args)
        for specs in ((), (SessionFoldSpec("protected", ("held_out",), ("visit_b",)),),
                      (SessionFoldSpec("one", ("visit_a",), ("visit_b",)),) * 2):
            with self.assertRaises(TrainingFoldError):
                create_session_folds(self.data, specs)

    def test_development_container_rejects_protected_metadata(self):
        changed = (replace(self.data.samples[0], partition=IdentityPartition.KNOWN_TEST), *self.data.samples[1:])
        with self.assertRaisesRegex(TrainingFoldError, "Protected"):
            DevelopmentDataset(self.data.X, self.data.y, changed, self.data.label_to_person, self.data.training_sessions)

    def test_development_container_rejects_missing_identity_sessions_and_wrong_labels(self):
        with self.assertRaisesRegex(TrainingFoldError, "Every development identity"):
            DevelopmentDataset(self.data.X[:8], self.data.y[:8], self.data.samples[:8],
                               self.data.label_to_person, self.data.training_sessions)
        wrong = self.data.y.copy()
        wrong[0] = 1
        with self.assertRaisesRegex(TrainingFoldError, "label"):
            DevelopmentDataset(self.data.X, wrong, self.data.samples, self.data.label_to_person, self.data.training_sessions)

    def test_invalid_row_indices_are_rejected(self):
        for indices in ((), (1, 0), (0, 0), (-1,), (len(self.data.samples),), (True,)):
            with self.subTest(case=indices), self.assertRaises(TrainingFoldError):
                TrainingRows(self.data, indices)
