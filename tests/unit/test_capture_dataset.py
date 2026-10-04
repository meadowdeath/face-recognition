from dataclasses import replace
from unittest.mock import Mock
import unittest

from face_recognition.application.capture_dataset import CaptureDataset, CapturePhase, INSTRUCTIONS
from face_recognition.domain.models.dataset import CONDITIONS, CaptureProtocol
from face_recognition.domain.models.face_landmarks import FaceLandmarks, Landmark
from face_recognition.domain.models.landmark_result import LandmarkResult
from tests.dataset_fakes import FakeDatasetRepository, raw_face, session_config


class CaptureDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = session_config()
        self.repository = FakeDatasetRepository(self.config)
        self.capture = CaptureDataset(self.repository, self.config)

    def result(self, timestamp: int, count: int = 2) -> LandmarkResult:
        return LandmarkResult((raw_face(count),), timestamp, 10)

    def warmup(self, start: int = 1) -> None:
        self.capture.start_condition()
        for timestamp in range(start, start + self.config.protocol.warmup_results):
            self.capture.observe(self.result(timestamp))

    def test_default_condition_saves_exactly_20_raw_time_spaced_samples(self) -> None:
        self.warmup()
        face = raw_face()
        for index in range(19):
            self.capture.observe(LandmarkResult((face,), 1000 + index * 250, 10))
        self.assertEqual(self.capture.accepted_count, 19)
        self.assertEqual(self.repository.writes, [])
        self.capture.observe(LandmarkResult((face,), 1000 + 19 * 250, 10))
        condition, samples = self.repository.writes[0]
        self.assertEqual(condition, "center")
        self.assertEqual(len(samples), 20)
        self.assertTrue(all(sample.landmarks is face for sample in samples))
        self.assertEqual([sample.sample_index for sample in samples], list(range(20)))
        self.assertEqual(self.capture.phase, CapturePhase.WAITING)
        self.assertEqual(self.capture.condition, "translation_left")
        self.assertEqual(self.capture.manifest.completed_conditions, ("center",))

    def test_warmup_is_not_persisted_and_duplicate_timestamps_never_count(self) -> None:
        self.capture.start_condition()
        for timestamp in range(1, 6):
            self.capture.observe(self.result(timestamp, 17))
            self.assertFalse(self.capture.observe(self.result(timestamp)))
        self.assertEqual(self.capture.warmup_count, 5)
        self.assertEqual(self.capture.accepted_count, 0)
        self.assertIsNone(self.capture.expected_landmark_count)
        self.capture.observe(self.result(1000))
        for _ in range(20):
            self.assertFalse(self.capture.observe(self.result(1000)))
        self.assertFalse(self.capture.observe(self.result(999)))
        self.assertEqual(self.capture.accepted_count, 1)
        self.assertEqual(self.repository.writes, [])

    def test_minimum_interval_accepts_boundary_and_skips_adjacent_results(self) -> None:
        self.warmup()
        for timestamp in (1000, 1001, 1100, 1249):
            self.capture.observe(self.result(timestamp))
        self.assertEqual(self.capture.accepted_count, 1)
        self.capture.observe(self.result(1250))
        self.assertEqual(self.capture.accepted_count, 2)
        self.assertEqual(self.capture.last_accepted_timestamp_ms, 1250)

    def test_missing_multiple_nonfinite_and_inconsistent_faces_are_not_accepted(self) -> None:
        self.warmup()
        self.capture.observe(self.result(1000))
        invalid = [LandmarkResult((), 1250, 10), LandmarkResult((raw_face(), raw_face()), 1500, 10),
                   self.result(1750, 3),
                   LandmarkResult((FaceLandmarks((Landmark(float("nan"), 0, 0), Landmark(0, 0, 0))),), 2000, 10)]
        for result in invalid:
            self.capture.observe(result)
            self.assertEqual(self.capture.accepted_count, 1)
            self.assertTrue(self.capture.status)
        self.capture.observe(self.result(2250))
        self.assertEqual(self.capture.accepted_count, 2)

    def test_no_face_warmup_completions_are_discarded(self) -> None:
        self.capture.start_condition()
        for timestamp in range(1, 6):
            self.capture.observe(LandmarkResult((), timestamp, 10))
        self.assertEqual(self.capture.phase, CapturePhase.COLLECTING)
        self.assertEqual(self.capture.accepted_count, 0)

    def test_waiting_snapshots_and_space_during_capture_do_not_reset_or_count(self) -> None:
        self.assertFalse(self.capture.observe(self.result(1)))
        self.capture.start_condition()
        self.assertFalse(self.capture.observe(self.result(1)))
        self.capture.observe(self.result(2))
        self.assertFalse(self.capture.start_condition())
        self.assertEqual(self.capture.warmup_count, 1)

    def test_cancellation_discards_unfinished_batch_without_repository_write(self) -> None:
        self.warmup()
        self.capture.observe(self.result(1000))
        self.capture.cancel()
        self.capture.observe(self.result(1250))
        self.assertEqual(self.capture.phase, CapturePhase.CANCELLED)
        self.assertEqual(self.repository.writes, [])

    def test_resume_skips_completed_conditions_and_keeps_established_count(self) -> None:
        repository = FakeDatasetRepository(self.config, ("center", "translation_left"))
        capture = CaptureDataset(repository, self.config)
        self.assertEqual(capture.condition, "translation_right")
        self.assertEqual(capture.expected_landmark_count, 2)
        capture.start_condition()
        for timestamp in range(1, 6):
            capture.observe(self.result(timestamp))
        capture.observe(self.result(1000, 3))
        self.assertEqual(capture.accepted_count, 0)

    def test_failed_persistence_does_not_advance_condition(self) -> None:
        self.repository.save_condition = Mock(side_effect=OSError("disk full"))
        self.warmup()
        for index in range(19):
            self.capture.observe(self.result(1000 + 250 * index))
        with self.assertRaisesRegex(OSError, "disk full"):
            self.capture.observe(self.result(1000 + 250 * 19))
        self.assertEqual(self.capture.stage_index, 0)
        self.assertEqual(self.capture.manifest.completed_conditions, ())

    def test_protocol_order_and_explicit_gates_for_all_13_conditions(self) -> None:
        expected = ("center", "translation_left", "translation_right", "translation_up", "translation_down",
                    "near", "far", "roll_left", "roll_right", "yaw_left", "yaw_right", "pitch_up", "pitch_down")
        self.assertEqual(CONDITIONS, expected)
        self.assertEqual(len(INSTRUCTIONS), 13)
        config = replace(self.config, protocol=CaptureProtocol(0, 1, 0))
        repository = FakeDatasetRepository(config)
        capture = CaptureDataset(repository, config)
        for timestamp, condition in enumerate(expected, start=1):
            self.assertEqual(capture.condition, condition)
            self.assertEqual(capture.phase, CapturePhase.WAITING)
            capture.start_condition()
            capture.observe(self.result(timestamp))
        self.assertEqual([condition for condition, _ in repository.writes], list(expected))
        self.assertEqual(capture.phase, CapturePhase.DONE)
        self.assertFalse(capture.start_condition())
