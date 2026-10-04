import unittest

from face_recognition.application.normalization_experiment import NormalizationExperiment
from face_recognition.domain.models.face_landmarks import FaceLandmarks, Landmark
from face_recognition.presentation.visualization.normalized_landmarks import NormalizedLandmarkRenderer


class NormalizedMappingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.renderer = NormalizedLandmarkRenderer()

    def test_fixed_range_and_origin(self) -> None:
        self.assertEqual(self.renderer.canvas_point(-2, -2), (0, 0))
        self.assertEqual(self.renderer.canvas_point(2, 2), (639, 639))
        self.assertEqual(self.renderer.canvas_point(0, 0), (320, 320))
        self.assertEqual(self.renderer.canvas_point(-0.5, 0), (240, 320))
        self.assertEqual(self.renderer.canvas_point(0.5, 0), (399, 320))

    def test_equal_scale_and_positive_y_down(self) -> None:
        origin = self.renderer.canvas_point(0, 0)
        x_pixel = self.renderer.canvas_point(1, 0)
        y_pixel = self.renderer.canvas_point(0, 1)
        self.assertEqual(x_pixel[0] - origin[0], y_pixel[1] - origin[1])
        self.assertGreater(y_pixel[1], origin[1])

    def test_outside_and_nonfinite_points_are_clipped_without_refitting(self) -> None:
        for x, y in ((2.01, 0), (0, -2.01), (float("nan"), 0), (0, float("inf"))):
            self.assertIsNone(self.renderer.canvas_point(x, y))
        self.assertEqual(self.renderer.canvas_point(0, 0), (320, 320))

    def test_rendering_keeps_coordinates_and_canvas_size_fixed(self) -> None:
        class IdentityNormalizer:
            def normalize(self, face: FaceLandmarks) -> FaceLandmarks:
                return face

        experiment = NormalizationExperiment(IdentityNormalizer(), 480, 270)
        face = FaceLandmarks((Landmark(-0.5, 0, 100), Landmark(0.5, 0, -100)))
        experiment.current_normalized = face
        original = face.points
        canvas = self.renderer.render(experiment)
        self.assertEqual(canvas.shape, (640, 640, 3))
        self.assertIs(face.points, original)
        self.assertEqual(face.points[0], Landmark(-0.5, 0, 100))


if __name__ == "__main__":
    unittest.main()
