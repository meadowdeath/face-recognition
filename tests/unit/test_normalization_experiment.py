from math import cos, sin, sqrt
import unittest
from unittest.mock import Mock

from face_recognition.application.normalization_experiment import (
    NormalizationExperiment, landmark_rmse, raw_comparison_geometry,
)
from face_recognition.domain.models.face_landmarks import FaceLandmarks, Landmark
from face_recognition.domain.models.landmark_result import LandmarkResult
from face_recognition.infrastructure.features.landmark_normalizer import LandmarkNormalizer


def synthetic_face(count: int = 363) -> FaceLandmarks:
    # Isotropic width-relative geometry encoded as MediaPipe coordinates at 480x270.
    points = [Landmark(0.5, 0.25 * 480 / 270, 0.02) for _ in range(count)]
    points[33] = points[133] = Landmark(0.4, 0.2 * 480 / 270, 0.01)
    points[362] = points[263] = Landmark(0.6, 0.2 * 480 / 270, 0.03)
    return FaceLandmarks(tuple(points))


class ComparisonTests(unittest.TestCase):
    def test_identical_landmarks_have_zero_rmse(self) -> None:
        face = synthetic_face()
        self.assertEqual(landmark_rmse(face, face), 0.0)

    def test_rmse_is_mean_3d_squared_displacement_per_landmark(self) -> None:
        baseline = FaceLandmarks((Landmark(0, 0, 0), Landmark(10, 20, 30)))
        current = FaceLandmarks((Landmark(3, 4, 12), Landmark(10, 20, 30)))
        self.assertAlmostEqual(landmark_rmse(current, baseline), 13 / sqrt(2))

    def test_raw_geometry_corrects_only_y_for_rectangular_images(self) -> None:
        raw = FaceLandmarks((Landmark(0.4, 0.8, -0.2),))
        corrected = raw_comparison_geometry(raw, 480, 270)
        self.assertEqual(corrected.points[0], Landmark(0.4, 0.45, -0.2))
        self.assertEqual(raw.points[0], Landmark(0.4, 0.8, -0.2))

    def test_raw_pixel_displacements_have_equal_x_y_z_scale(self) -> None:
        baseline = FaceLandmarks((Landmark(100 / 480, 100 / 270, 10 / 480),))
        current = FaceLandmarks((Landmark(130 / 480, 140 / 270, 130 / 480),))
        error = landmark_rmse(raw_comparison_geometry(current, 480, 270),
                              raw_comparison_geometry(baseline, 480, 270))
        self.assertAlmostEqual(error, 130 / 480)  # sqrt(30² + 40² + 120²) / width
        self.assertNotAlmostEqual(landmark_rmse(current, baseline), error)

    def test_mismatched_counts_are_explicitly_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Landmark counts must match"):
            landmark_rmse(synthetic_face(364), synthetic_face(363))

    def test_raw_geometry_requires_positive_dimensions(self) -> None:
        for dimensions in ((0, 270), (480, 0), (-1, 270)):
            with self.subTest(dimensions=dimensions):
                with self.assertRaisesRegex(ValueError, "positive"):
                    raw_comparison_geometry(synthetic_face(), *dimensions)


class ExperimentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.normalizer = Mock(wraps=LandmarkNormalizer(480, 270))
        self.experiment = NormalizationExperiment(self.normalizer, 480, 270)
        self.face = synthetic_face()

    def observe(self, face: FaceLandmarks | None, timestamp: int) -> bool:
        return self.experiment.observe(LandmarkResult(() if face is None else (face,), timestamp, 10.0))

    def test_reused_and_older_results_do_not_recompute_or_count(self) -> None:
        self.assertTrue(self.observe(self.face, 100))
        self.assertFalse(self.observe(synthetic_face(364), 100))
        self.assertFalse(self.observe(None, 99))
        self.assertFalse(self.experiment.observe(None))
        self.assertEqual(self.experiment.observations, 1)
        self.normalizer.normalize.assert_called_once_with(self.face)
        self.assertIs(self.experiment.current_raw, self.face)
        self.assertTrue(self.observe(self.face, 101))
        self.assertEqual(self.experiment.observations, 2)
        self.assertEqual(self.normalizer.normalize.call_count, 2)

    def test_reference_stores_raw_and_normalized_current_face(self) -> None:
        self.assertFalse(self.experiment.capture_reference())
        self.observe(self.face, 100)
        normalized = self.experiment.current_normalized
        self.assertTrue(self.experiment.capture_reference())
        self.assertIs(self.experiment.reference.raw, self.face)
        self.assertIs(self.experiment.reference.normalized, normalized)
        self.assertEqual(self.experiment.raw_rmse, 0.0)
        self.assertEqual(self.experiment.normalized_rmse, 0.0)
        self.normalizer.normalize.assert_called_once()

    def test_no_face_completion_clears_current_but_preserves_reference(self) -> None:
        self.observe(self.face, 100)
        self.experiment.capture_reference()
        reference = self.experiment.reference
        self.assertTrue(self.observe(None, 101))
        self.assertIsNone(self.experiment.current_raw)
        self.assertIsNone(self.experiment.current_normalized)
        self.assertIsNone(self.experiment.raw_rmse)
        self.assertFalse(self.experiment.capture_reference())
        self.assertIs(self.experiment.reference, reference)
        self.assertEqual(self.experiment.observations, 2)

    def test_normalization_failure_does_not_replace_reference(self) -> None:
        self.observe(self.face, 100)
        self.experiment.capture_reference()
        reference = self.experiment.reference
        self.observe(FaceLandmarks((Landmark(0, 0, 0),)), 101)
        self.assertIn("indices", self.experiment.error)
        self.assertIsNone(self.experiment.normalized_rmse)
        self.assertFalse(self.experiment.capture_reference())
        self.assertIs(self.experiment.reference, reference)

    def test_mismatched_counts_report_unavailable_and_allow_new_reference(self) -> None:
        self.observe(self.face, 100)
        self.experiment.capture_reference()
        changed = synthetic_face(364)
        self.observe(changed, 101)
        self.assertIn("Landmark counts must match", self.experiment.error)
        self.assertIsNone(self.experiment.raw_rmse)
        self.assertIsNone(self.experiment.normalized_rmse)
        self.assertTrue(self.experiment.capture_reference())
        self.assertIs(self.experiment.reference.raw, changed)
        self.assertEqual(self.experiment.raw_rmse, 0.0)
        self.assertIsNone(self.experiment.error)

    def test_raw_baseline_retains_movement_while_normalization_removes_it(self) -> None:
        self.observe(self.face, 100)
        self.experiment.capture_reference()
        geometry = raw_comparison_geometry(self.face, 480, 270)
        for timestamp, (scale, roll, tx, ty, tz) in enumerate(
            ((1, 0, 0.1, -0.03, 0.04), (1.2, 0, 0, 0, 0),
             (1, 0.3, 0, 0, 0), (1.3, -0.4, 0.05, 0.03, -0.02)), start=101,
        ):
            with self.subTest(timestamp=timestamp):
                transformed = FaceLandmarks(tuple(
                    Landmark(scale * (cos(roll) * p.x - sin(roll) * p.y) + tx,
                             (scale * (sin(roll) * p.x + cos(roll) * p.y) + ty) * 480 / 270,
                             scale * p.z + tz) for p in geometry.points
                ))
                self.observe(transformed, timestamp)
                self.assertGreater(self.experiment.raw_rmse, 0.01)
                self.assertAlmostEqual(self.experiment.normalized_rmse, 0, delta=1e-12)


if __name__ == "__main__":
    unittest.main()
