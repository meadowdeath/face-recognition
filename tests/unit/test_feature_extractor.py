from dataclasses import FrozenInstanceError
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


    def test_flattening_preserves_landmark_index_and_xyz_order(self) -> None:
        landmarks = FaceLandmarks((Landmark(1, 2, 3), Landmark(4, 5, 6)))
        extractor: FeatureExtractor = LandmarkFeatureExtractor()
        features = extractor.extract(landmarks)
        self.assertIsInstance(features, FaceFeatures)
        self.assertEqual(features.values, (1, 2, 3, 4, 5, 6))

    def test_feature_count_supports_arbitrary_landmark_counts(self) -> None:
        for count in (1, 2, 17, 363, 478, 500):
            with self.subTest(count=count):
                points = tuple(Landmark(-index - 0.5, index + 0.25, index - 0.125)
                               for index in range(count))
                features = LandmarkFeatureExtractor().extract(FaceLandmarks(points))
                self.assertEqual(len(features.values), count * 3)
                for index, point in enumerate(points):
                    self.assertEqual(features.values[index * 3:index * 3 + 3],
                                     (point.x, point.y, point.z))

    def test_extraction_is_deterministic(self) -> None:
        landmarks = FaceLandmarks((Landmark(-0.5, 0.25, -0.125), Landmark(0.5, 0, 0.1)))
        extractor = LandmarkFeatureExtractor()
        self.assertEqual(extractor.extract(landmarks), extractor.extract(landmarks))

    def test_input_is_not_modified(self) -> None:
        points = (Landmark(-0.5, 0.25, 0.125), Landmark(0.5, -0.25, -0.125))
        landmarks = FaceLandmarks(points)
        original_values = tuple((point.x, point.y, point.z) for point in points)
        LandmarkFeatureExtractor().extract(landmarks)
        self.assertIs(landmarks.points, points)
        self.assertEqual(tuple((point.x, point.y, point.z) for point in landmarks.points), original_values)

    def test_returned_feature_values_are_immutable(self) -> None:
        features = LandmarkFeatureExtractor().extract(FaceLandmarks((Landmark(1, 2, 3),)))
        self.assertIsInstance(features.values, tuple)
        with self.assertRaises(TypeError):
            features.values[0] = 99
        with self.assertRaises(FrozenInstanceError):
            features.values = (99,)

    def test_nonfinite_values_are_rejected_in_every_coordinate(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            for point_index in (0, 1):
                for coordinate_index, coordinate in enumerate(("x", "y", "z")):
                    with self.subTest(value=value, point=point_index, coordinate=coordinate):
                        coordinates = [1.0, 2.0, 3.0]
                        coordinates[coordinate_index] = value
                        points = [Landmark(0, 0, 0), Landmark(0, 0, 0)]
                        points[point_index] = Landmark(*coordinates)
                        with self.assertRaisesRegex(ValueError, "coordinates must be finite"):
                            LandmarkFeatureExtractor().extract(FaceLandmarks(tuple(points)))
