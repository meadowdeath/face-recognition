from dataclasses import replace
import unittest

from face_recognition.domain.models.dataset import CaptureProtocol, RawLandmarkSample, validate_identifier
from face_recognition.domain.models.face_landmarks import FaceLandmarks, Landmark
from tests.dataset_fakes import condition_samples, raw_face, session_config


class DatasetModelTests(unittest.TestCase):
    def test_safe_pseudonymous_identifiers(self) -> None:
        for value in ("p001", "person-2", "session_01", "A_1-2"):
            self.assertEqual(validate_identifier(value), value)

    def test_unsafe_and_windows_reserved_identifiers_are_rejected(self) -> None:
        for value in ("", "..", "../p001", "p/1", "p\\1", "C:folder", "p.1", "full name", "ñ", "a" * 65,
                      "CON", "nul", "COM1", "LPT9"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_identifier(value)
        for field in ("person_id", "session_id"):
            with self.assertRaises(ValueError):
                session_config(**{field: "../outside"})

    def test_protocol_defaults_and_validation(self) -> None:
        self.assertEqual(CaptureProtocol(), CaptureProtocol(5, 20, 250))
        for args in ((-1, 20, 250), (5, 0, 250), (5, 20, -1), (True, 20, 250), (5, 20.0, 250)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                CaptureProtocol(*args)

    def test_arbitrary_landmark_counts_and_nonempty_requirement(self) -> None:
        for count in (1, 2, 17, 478, 500):
            sample = RawLandmarkSample("p001", "session_01", "center", 0, 1000, 480, 270, raw_face(count))
            self.assertEqual(len(sample.landmarks.points), count)
        with self.assertRaises(ValueError):
            FaceLandmarks(())

    def test_nonfinite_coordinates_are_rejected_at_every_axis(self) -> None:
        sample = condition_samples(session_config())[0]
        for value in (float("nan"), float("inf"), float("-inf")):
            for axis in range(3):
                coordinates = [0.1, 0.2, 0.3]
                coordinates[axis] = value
                with self.subTest(value=value, axis=axis), self.assertRaisesRegex(ValueError, "finite"):
                    replace(sample, landmarks=FaceLandmarks((Landmark(*coordinates),)))

    def test_invalid_sample_metadata_is_rejected(self) -> None:
        sample = condition_samples(session_config())[0]
        for fields in ({"condition": "../center"}, {"sample_index": -1}, {"detector_timestamp_ms": -1},
                       {"inference_width": 0}, {"inference_height": True}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                replace(sample, **fields)
