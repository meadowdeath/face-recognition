from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from face_recognition.application.final_evaluation_data import FinalEvaluationError
from face_recognition.domain.models.identity_split import IdentityPartition
from face_recognition.infrastructure.persistence.final_holdout_loader import FinalHoldoutLoader, validate_final_provenance
from face_recognition.infrastructure.persistence.open_set_config_repository import read_aggregate_development_report, read_frozen_configuration
from face_recognition.infrastructure.persistence.model_repository import load_open_set_model
from tests.final_evaluation_fakes import final_fixture


class FinalHoldoutLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory(prefix="synthetic_final_reader_")
        cls.bundle = final_fixture(Path(cls.temporary.name))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_reader_opens_exact_final_whitelist_and_never_development_data(self):
        fixture, _, _ = self.bundle
        processed, source, partition, _, _, _ = fixture
        forbidden = {processed / s.source_relative_path for s in source.samples
                     if s.partition is IdentityPartition.KNOWN_TRAIN_POOL or s.person_id in partition.specification.calibration_ids}
        allowed = {processed / s.source_relative_path for s in source.samples
                   if s.partition is IdentityPartition.KNOWN_TEST or s.person_id in partition.specification.holdout_ids}
        original, opened = Path.open, set()

        def guard(path, *args, **kwargs):
            self.assertNotIn(path, forbidden)
            opened.add(path)
            return original(path, *args, **kwargs)

        with patch.object(Path, "open", guard):
            data = FinalHoldoutLoader(processed).load_final(source, partition, source.specification.known_ids, 1434)
        self.assertEqual({p for p in opened if p.suffix == ".jsonl"}, allowed)
        self.assertFalse(data.known_X.flags.writeable or data.unknown_X.flags.writeable)
        wrong = next(s for s in source.samples if s.partition is IdentityPartition.KNOWN_TRAIN_POOL)
        with self.assertRaises(FinalEvaluationError):
            replace(data, known_samples=(wrong, *data.known_samples[1:]))

    def test_digest_bound_unknown_assignment_validated_before_payloads(self):
        fixture, development_path, model_dir = self.bundle
        _, source, partition, _, _, config_path = fixture
        frozen = read_frozen_configuration(config_path)
        model = load_open_set_model(model_dir, frozen)
        development = read_aggregate_development_report(development_path)
        validate_final_provenance(model, frozen, source, partition, development)
        with self.assertRaises(FinalEvaluationError):
            validate_final_provenance(model, frozen, source, partition, replace(development, sha256="f" * 64))
        with self.assertRaises(FinalEvaluationError):
            FinalHoldoutLoader(fixture[0]).load_final(source, partition, tuple(reversed(model.parts.class_ids)), 1434)

    def test_wrong_feature_dimensions_fail_explicitly(self):
        fixture = self.bundle[0]
        with self.assertRaises(FinalEvaluationError):
            FinalHoldoutLoader(fixture[0]).load_final(fixture[1], fixture[2], fixture[1].specification.known_ids, 448)
