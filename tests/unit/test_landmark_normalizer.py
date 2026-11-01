from math import cos, pi, sin, sqrt
import unittest

from face_recognition.domain.models.face_landmarks import FaceLandmarks, Landmark
from face_recognition.infrastructure.features.landmark_normalizer import LandmarkNormalizer


class LandmarkNormalizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.normalizer = LandmarkNormalizer(480, 270)
        self.pixel_face = self.make_face()
        self.face = self.to_image_normalized(self.pixel_face)
        self.expected = self.normalizer.normalize(self.face)

    @staticmethod
    def make_face(count: int = 400) -> FaceLandmarks:
        # Geometry is defined in isotropic pixel-equivalent units, including z.
        points = [Landmark(180 + (index % 17) * 4, 70 + (index % 13) * 5,
                           -10 + (index % 11) * 3) for index in range(count)]
        # Eye centers (200,80,30), (280,160,50): origin (240,120,40), d=80*sqrt(2).
        points[33], points[133] = Landmark(180, 80, 20), Landmark(220, 80, 40)
        points[362], points[263] = Landmark(260, 160, 40), Landmark(300, 160, 60)
        points[0] = Landmark(240, 120, 40)
        points[1] = Landmark(280, 160, 40 + 80 * sqrt(2))
        points[2] = Landmark(240, 200, 40 - 80 * sqrt(2))
        points[3] = Landmark(320, 40, 40)
        return FaceLandmarks(tuple(points))

    @staticmethod
    def to_image_normalized(
        face: FaceLandmarks, width: int = 480, height: int = 270,
    ) -> FaceLandmarks:
        # Emulate MediaPipe's different scales for x and y; z follows width.
        return FaceLandmarks(tuple(Landmark(point.x / width, point.y / height, point.z / width)
                                   for point in face.points))

    @staticmethod
    def transform(
        face: FaceLandmarks, *, translation: tuple[float, float, float] = (0, 0, 0),
        scale: float = 1.0, roll: float = 0.0,
    ) -> FaceLandmarks:
        cosine, sine = cos(roll), sin(roll)
        tx, ty, tz = translation
        return FaceLandmarks(tuple(
            Landmark(scale * (cosine * point.x - sine * point.y) + tx,
                     scale * (sine * point.x + cosine * point.y) + ty,
                     scale * point.z + tz)
            for point in face.points
        ))

    def assert_landmarks_close(self, actual: FaceLandmarks, expected: FaceLandmarks) -> None:
        self.assertEqual(len(actual.points), len(expected.points))
        for index, (point, reference) in enumerate(zip(actual.points, expected.points)):
            for coordinate in ("x", "y", "z"):
                self.assertAlmostEqual(getattr(point, coordinate), getattr(reference, coordinate),
                                       delta=1e-10, msg=f"landmark {index}, {coordinate}")

    def test_translation_invariance_including_z(self) -> None:
        for translation in ((20.0, -15.0, 5.3), (-10.0, 20.0, -30.0)):
            with self.subTest(translation=translation):
                moved = self.to_image_normalized(self.transform(self.pixel_face, translation=translation))
                self.assert_landmarks_close(self.normalizer.normalize(moved), self.expected)

    def test_positive_uniform_scale_invariance(self) -> None:
        for scale in (0.1, 2.5, 17.0):
            with self.subTest(scale=scale):
                scaled = self.to_image_normalized(self.transform(self.pixel_face, scale=scale))
                self.assert_landmarks_close(self.normalizer.normalize(scaled), self.expected)

    def test_roll_invariance_across_eye_axis_angles(self) -> None:
        for roll in (-pi, 0.03, 0.7, pi / 2, 2.4):
            with self.subTest(roll=roll):
                rotated = self.to_image_normalized(self.transform(self.pixel_face, roll=roll))
                self.assert_landmarks_close(self.normalizer.normalize(rotated), self.expected)

    def test_combined_translation_scale_and_roll_invariance(self) -> None:
        for roll in (-0.8, 1.4, pi - 0.1):
            with self.subTest(roll=roll):
                changed = self.to_image_normalized(
                    self.transform(self.pixel_face, translation=(20.0, -70.0, 50.0), scale=3.2, roll=roll)
                )
                self.assert_landmarks_close(self.normalizer.normalize(changed), self.expected)

    def test_insufficient_reference_landmarks_are_rejected(self) -> None:
        for count in (1, 100, 362):
            with self.subTest(count=count):
                face = FaceLandmarks(tuple(Landmark(0, 0, 0) for _ in range(count)))
                with self.assertRaisesRegex(ValueError, "indices 33, 133, 362, and 263"):
                    self.normalizer.normalize(face)

    def test_zero_and_near_zero_2d_interocular_distance_are_rejected(self) -> None:
        for distance in (0.0, 1e-12, 1e-9, 1e-8):
            with self.subTest(distance=distance):
                points = list(self.face.points)
                points[33] = points[133] = Landmark(0, 0, -100)
                points[362] = points[263] = Landmark(distance, 0, 100)
                with self.assertRaisesRegex(ValueError, "Interocular distance"):
                    self.normalizer.normalize(FaceLandmarks(tuple(points)))

    def test_eye_centers_and_known_points_have_expected_xyz_coordinates(self) -> None:
        points = self.expected.points
        for first, second, x in ((33, 133, -0.5), (362, 263, 0.5)):
            self.assertAlmostEqual((points[first].x + points[second].x) / 2, x, delta=1e-12)
            self.assertAlmostEqual((points[first].y + points[second].y) / 2, 0, delta=1e-12)
        self.assert_landmarks_close(
            FaceLandmarks(points[:4]),
            FaceLandmarks((Landmark(0, 0, 0), Landmark(0.5, 0, 1),
                           Landmark(0.5, 0.5, -1), Landmark(0, -1, 0))),
        )
        self.assertAlmostEqual(sum(points[index].z for index in (33, 133, 362, 263)) / 4, 0)

    def test_number_of_landmarks_preserved_without_requiring_478(self) -> None:
        for count in (363, 400, 478, 500):
            with self.subTest(count=count):
                normalized = self.normalizer.normalize(self.to_image_normalized(self.make_face(count)))
                self.assertEqual(len(normalized.points), count)

    def test_input_is_unchanged_and_output_is_a_new_instance(self) -> None:
        original_points = self.face.points
        original_values = tuple((point.x, point.y, point.z) for point in original_points)
        normalized = self.normalizer.normalize(self.face)
        self.assertIsNot(normalized, self.face)
        self.assertIsNot(normalized.points, original_points)
        self.assertIs(self.face.points, original_points)
        self.assertEqual(tuple((point.x, point.y, point.z) for point in self.face.points), original_values)
        self.assertTrue(all(new is not old for new, old in zip(normalized.points, original_points)))

    def test_invalid_image_dimensions_are_rejected(self) -> None:
        for width, height in ((0, 270), (480, 0), (-480, 270), (480, -270),
                              (0, 0), (480.0, 270), (True, 270), (480, "270")):
            with self.subTest(width=width, height=height):
                with self.assertRaisesRegex(ValueError, "Image dimensions must be positive integers"):
                    LandmarkNormalizer(width, height)

    def test_same_pixel_geometry_normalizes_identically_across_aspect_ratios(self) -> None:
        for width, height in ((480, 270), (270, 480), (640, 480), (480, 480)):
            with self.subTest(width=width, height=height):
                encoded = self.to_image_normalized(self.pixel_face, width, height)
                normalized = LandmarkNormalizer(width, height).normalize(encoded)
                self.assert_landmarks_close(normalized, self.expected)

    def test_rectangular_geometry_requires_the_correct_dimensions(self) -> None:
        # Treating 480x270 encoding as square reproduces the original distortion.
        distorted = LandmarkNormalizer(480, 480).normalize(self.face)
        self.assertAlmostEqual(self.expected.points[3].x, 0.0, delta=1e-12)
        self.assertGreater(abs(distorted.points[3].x), 0.1)
        self.assertAlmostEqual(self.expected.points[3].y, -1.0, delta=1e-12)
