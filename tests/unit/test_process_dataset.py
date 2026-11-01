from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock
import unittest

from face_recognition.application.process_dataset import DatasetProcessor, DatasetProcessingError
from face_recognition.domain.models.face_landmarks import FaceLandmarks, Landmark
from face_recognition.infrastructure.features.landmark_feature_extractor import LandmarkFeatureExtractor
from face_recognition.infrastructure.features.landmark_normalizer import LandmarkNormalizer
from face_recognition.infrastructure.persistence.raw_dataset_reader import RawDatasetReader
from tests.processed_dataset_fakes import MemoryProcessedRepository, synthetic_sample, write_synthetic_session


class DatasetProcessorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.processor = DatasetProcessor(LandmarkNormalizer, LandmarkFeatureExtractor())

    def test_single_sample_reuses_normalizer_extractor_and_preserves_provenance(self) -> None:
        sample = synthetic_sample()
        original = sample.landmarks.points
        normalized, baseline = self.processor.process_sample(sample)
        expected = LandmarkNormalizer(sample.inference_width, sample.inference_height).normalize(sample.landmarks)
        expected_features = LandmarkFeatureExtractor().extract(expected)
        self.assertTrue(normalized.landmarks == expected, "Normalization must exactly reuse the existing algorithm")
        self.assertTrue(baseline.features == expected_features, "Feature flattening must exactly reuse the existing extractor")
        self.assertEqual(normalized.provenance, baseline.provenance)
        provenance = normalized.provenance
        self.assertEqual((provenance.person_id, provenance.session_id, provenance.condition, provenance.sample_index),
                         (sample.person_id, sample.session_id, sample.condition, sample.sample_index))
        self.assertEqual(provenance.source_relative_path, "subject_test/visit_test/center.jsonl")
        self.assertIs(sample.landmarks.points, original)

    def test_each_sample_uses_its_own_dimensions(self) -> None:
        factory = Mock(side_effect=LandmarkNormalizer)
        processor = DatasetProcessor(factory, LandmarkFeatureExtractor())
        for width, height in ((480, 270), (270, 480), (640, 480)):
            sample = synthetic_sample(width=width, height=height)
            actual, _ = processor.process_sample(sample)
            expected = LandmarkNormalizer(width, height).normalize(sample.landmarks)
            self.assertTrue(actual.landmarks == expected, "Use source dimensions, not current camera settings")
        self.assertEqual([call.args for call in factory.call_args_list], [(480, 270), (270, 480), (640, 480)])

    def test_counts_and_coordinate_order_are_preserved_for_arbitrary_counts(self) -> None:
        for count in (363, 400, 478, 500):
            with self.subTest(count=count):
                normalized, baseline = self.processor.process_sample(synthetic_sample(count=count))
                self.assertEqual(len(normalized.landmarks.points), count)
                self.assertEqual(len(baseline.features.values), count * 3)
                for index, point in enumerate(normalized.landmarks.points):
                    self.assertTrue(baseline.features.values[index * 3:index * 3 + 3] == (point.x, point.y, point.z),
                                    "Feature ordering must remain landmark index then X/Y/Z")

    def test_normalization_error_has_sample_context_but_no_underlying_values(self) -> None:
        sample = synthetic_sample(index=3)
        points = list(sample.landmarks.points)
        for index in (33, 133, 362, 263):
            points[index] = Landmark(0, 0, 0)
        invalid = replace(sample, landmarks=FaceLandmarks(tuple(points)))
        with self.assertRaises(DatasetProcessingError) as caught:
            self.processor.process_sample(invalid)
        error = caught.exception
        self.assertEqual(error.issue.category, "normalization_failed")
        for text in ("subject_test", "visit_test", "center", "sample_index=3"):
            self.assertIn(text, str(error))
        self.assertTrue(error.__suppress_context__)

    def test_nonfinite_input_is_rejected_even_for_a_tampered_in_memory_sample(self) -> None:
        # Normal construction rejects this already; defend the application boundary too.
        sample = synthetic_sample()
        object.__setattr__(sample, "landmarks", FaceLandmarks((Landmark(float("nan"), 0, 0),)))
        with self.assertRaises(DatasetProcessingError) as caught:
            self.processor.process_sample(sample)
        self.assertEqual(caught.exception.issue.category, "input_validation_failed")

    def test_injected_errors_never_echo_exception_contents(self) -> None:
        marker = "SYNTHETIC_BIOMETRIC_ERROR_TEXT"
        factory = Mock(side_effect=ValueError(marker))
        with self.assertRaises(DatasetProcessingError) as caught:
            DatasetProcessor(factory, LandmarkFeatureExtractor()).process_sample(synthetic_sample())
        self.assertNotIn(marker, str(caught.exception))
        extractor = Mock()
        extractor.extract.side_effect = ValueError(marker)
        with self.assertRaises(DatasetProcessingError) as caught:
            DatasetProcessor(LandmarkNormalizer, extractor).process_sample(synthetic_sample())
        self.assertEqual(caught.exception.issue.category, "feature_extraction_failed")
        self.assertNotIn(marker, str(caught.exception))


class DatasetProcessorReaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory(prefix="synthetic_processing_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "raw"
        self.root.mkdir()
        self.processor = DatasetProcessor(LandmarkNormalizer, LandmarkFeatureExtractor())

    def test_reader_integration_is_deterministic_with_correct_counts(self) -> None:
        for person, session in (("zeta_test", "visit_b"), ("alpha_test", "visit_z"), ("alpha_test", "visit_a")):
            write_synthetic_session(self.root, person, session)
        repository = MemoryProcessedRepository()
        summary = self.processor.process(RawDatasetReader(self.root), repository)
        self.assertEqual((summary.participants, summary.sessions, summary.conditions, summary.total_samples_processed),
                         (2, 3, 6, 12))
        self.assertEqual((summary.error_count, summary.rejected_samples), (0, 0))
        keys = [(n.provenance.person_id, n.provenance.session_id, n.provenance.condition, n.provenance.sample_index)
                for n, _ in repository.samples]
        expected = [(person, session, condition, index)
                    for person, session in (("alpha_test", "visit_a"), ("alpha_test", "visit_z"), ("zeta_test", "visit_b"))
                    for condition in ("center", "translation_left") for index in range(2)]
        self.assertEqual(keys, expected)

    def test_failure_stops_processing_and_reports_rejection_without_publishing(self) -> None:
        directory = write_synthetic_session(self.root)
        path = directory / "center.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        data = json.loads(lines[1])
        for index in (33, 133, 362, 263):
            data["landmarks"][index] = [0, 0, 0]
        lines[1] = json.dumps(data)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        repository = MemoryProcessedRepository()
        with self.assertRaises(DatasetProcessingError) as caught:
            self.processor.process(RawDatasetReader(self.root), repository)
        self.assertEqual(len(repository.samples), 1)
        self.assertIsNone(repository.summary)
        summary, issue = repository.failure
        self.assertEqual((summary.total_samples_processed, summary.rejected_samples, summary.error_count), (1, 1, 1))
        self.assertEqual(issue.sample_index, 1)
        self.assertEqual(caught.exception.issue, issue)

    def test_reader_nonfinite_failure_is_explicit_and_has_file_context(self) -> None:
        directory = write_synthetic_session(self.root)
        path = directory / "center.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        data = json.loads(lines[0])
        data["landmarks"][0][0] = float("inf")
        lines[0] = json.dumps(data)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        repository = MemoryProcessedRepository()
        with self.assertRaises(DatasetProcessingError) as caught:
            self.processor.process(RawDatasetReader(self.root), repository)
        issue = caught.exception.issue
        self.assertEqual(issue.category, "source_validation_failed")
        self.assertEqual((issue.person_id, issue.session_id, issue.condition), ("subject_test", "visit_test", "center"))
        self.assertEqual(issue.sample_index, 0)
        self.assertEqual(caught.exception.summary.rejected_samples, 1)
        self.assertEqual(len(repository.samples), 0)
        self.assertIsNone(repository.summary)
