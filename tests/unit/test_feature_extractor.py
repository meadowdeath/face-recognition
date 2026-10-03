import unittest

from face_recognition.domain.interfaces.feature_extractor import FeatureExtractor
from face_recognition.domain.models.face_features import FaceFeatures
from face_recognition.domain.models.face_landmarks import FaceLandmarks, Landmark
from face_recognition.infrastructure.features.landmark_feature_extractor import LandmarkFeatureExtractor


class FeatureExtractorTests(unittest.TestCase):
    def test_feature_extractor_contract_accepts_an_implementation(self) -> None:
        class FakeExtractor:
            def extract(self, landmarks: FaceLandmarks) -> FaceFeatures:
                return FaceFeatures((landmarks.points[0].x,))

        extractor: FeatureExtractor = FakeExtractor()
        self.assertEqual(extractor.extract(FaceLandmarks((Landmark(0.3, 0.4, 0.0),))).values, (0.3,))


    def test_real_extractor_explicitly_waits_for_feature_selection(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "Select and normalize"):
            LandmarkFeatureExtractor().extract(FaceLandmarks((Landmark(0, 0, 0),)))
