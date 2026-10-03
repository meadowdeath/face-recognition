import unittest

from face_recognition.domain.models.face_features import FaceFeatures
from face_recognition.domain.models.face_landmarks import FaceLandmarks, Landmark
from face_recognition.domain.models.prediction import Prediction


class DomainModelTests(unittest.TestCase):
    def test_domain_values(self) -> None:
        face = FaceLandmarks((Landmark(0.1, 0.2, -0.01),))
        self.assertEqual(face.points[0].x, 0.1)
        self.assertEqual(FaceFeatures((1.0, 2.0)).values, (1.0, 2.0))
        self.assertIsNone(Prediction(None).label)


    def test_empty_landmarks_and_features_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            FaceLandmarks(())
        with self.assertRaises(ValueError):
            FaceFeatures(())
