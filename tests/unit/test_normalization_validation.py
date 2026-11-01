from math import sqrt
import unittest
from unittest.mock import Mock

from face_recognition.application.normalization_validation import (
    CONDITIONS, STAGES, NormalizationValidation, Phase, ValidationConfig,
    mean_landmark_template, summarize_rmse,
)
from face_recognition.domain.models.face_landmarks import FaceLandmarks, Landmark
from face_recognition.domain.models.landmark_result import LandmarkResult
from face_recognition.infrastructure.features.landmark_normalizer import LandmarkNormalizer


def face(offset: float = 0, count: int = 363) -> FaceLandmarks:
    points = [Landmark(0.5 + offset, 0.5, 0.02) for _ in range(count)]
    points[33] = points[133] = Landmark(0.4 + offset, 0.4, 0.01)
    points[362] = points[263] = Landmark(0.6 + offset, 0.4, 0.03)
    return FaceLandmarks(tuple(points))


class TemplateStatisticsTests(unittest.TestCase):
    def test_mean_template_averages_every_coordinate_at_corresponding_indices(self) -> None:
        samples = [FaceLandmarks((Landmark(1, 4, -3), Landmark(10, 20, 30))),
                   FaceLandmarks((Landmark(3, 8, 1), Landmark(20, 40, 60))),
                   FaceLandmarks((Landmark(8, 12, 8), Landmark(30, 60, 90)))]
        expected = FaceLandmarks((Landmark(4, 8, 2), Landmark(20, 40, 60)))
        self.assertEqual(mean_landmark_template(samples), expected)
        self.assertEqual(samples[0].points[0], Landmark(1, 4, -3))

    def test_template_rejects_empty_samples_and_mismatched_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            mean_landmark_template([])
        with self.assertRaisesRegex(ValueError, "counts must match"):
            mean_landmark_template([face(), face(count=364)])

    def test_statistics_use_population_standard_deviation(self) -> None:
        stats = summarize_rmse([1, 2, 3, 4])
        self.assertEqual(stats.mean, 2.5)
        self.assertAlmostEqual(stats.std, sqrt(1.25))
        self.assertEqual(stats.median, 2.5)
        self.assertEqual(stats.minimum, 1)
        self.assertEqual(stats.maximum, 4)

    def test_single_value_has_zero_population_standard_deviation(self) -> None:
        stats = summarize_rmse([0.125])
        self.assertEqual(stats.mean, stats.median)
        self.assertEqual(stats.std, 0)

    def test_statistics_reject_empty_and_nonfinite_values(self) -> None:
        for values in ([], [float("nan")], [float("inf")]):
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, "finite"):
                    summarize_rmse(values)

    def test_sampling_configuration_validation(self) -> None:
        for warmup, samples in ((-1, 30), (5, 0), (1.5, 30), (5, True)):
            with self.subTest(warmup=warmup, samples=samples):
                with self.assertRaises(ValueError):
                    ValidationConfig(warmup, samples)


class ValidationSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.normalizer = Mock(wraps=LandmarkNormalizer(480, 270))
        self.session = NormalizationValidation(self.normalizer, 480, 270)
        self.timestamp = 0

    def submit(self, sample: FaceLandmarks | None = None) -> None:
        self.timestamp += 1
        self.session.observe(LandmarkResult(() if sample is None else (sample,), self.timestamp, 10))

    def complete_stage(self, offset: float = 0) -> None:
        self.assertTrue(self.session.start_stage())
        for _ in range(self.session.config.warmup_results):
            self.submit(face(10))  # Warm-up outliers must never affect templates/statistics.
        for _ in range(self.session.config.measurement_results):
            self.submit(face(offset))

    def test_default_counts_and_reference_then_center_are_independent(self) -> None:
        self.complete_stage()
        self.assertEqual(self.session.config, ValidationConfig(5, 30))
        self.assertEqual(self.session.stage_index, 1)
        self.assertEqual(self.session.phase, Phase.WAITING)
        self.assertEqual(self.session.reports, [])
        reference = self.session.reference
        self.assertAlmostEqual(reference.raw.points[0].x, 0.5)
        self.assertAlmostEqual(reference.raw.points[0].y, 0.5 * 270 / 480)
        self.complete_stage(0.1)
        self.assertIs(self.session.reference, reference)
        self.assertEqual(len(self.session.reports), 1)
        report = self.session.reports[0]
        self.assertEqual(report.condition.name, "CENTER")
        self.assertEqual(report.samples, 30)
        self.assertAlmostEqual(report.raw.mean, 0.1)
        self.assertAlmostEqual(report.raw.std, 0)
        self.assertAlmostEqual(report.normalized.mean, 0, delta=1e-12)

    def test_reference_templates_are_means_not_a_single_reference_frame(self) -> None:
        self.session = NormalizationValidation(self.normalizer, 480, 270, ValidationConfig(0, 2))
        self.session.start_stage()
        first, second = face(0), face(0.2)
        expected_raw_x = (first.points[0].x + second.points[0].x) / 2
        expected_normalized_z = (self.normalizer.normalize(first).points[0].z
                                 + self.normalizer.normalize(second).points[0].z) / 2
        self.submit(first)
        self.submit(second)
        self.assertAlmostEqual(self.session.reference.raw.points[0].x, expected_raw_x)
        self.assertAlmostEqual(self.session.reference.normalized.points[0].z, expected_normalized_z)

    def test_exactly_30_valid_new_results_complete_reference_after_warmup(self) -> None:
        self.session.start_stage()
        for _ in range(5):
            self.submit(face())
        self.assertEqual(self.session.phase, Phase.COLLECTING)
        self.assertEqual(self.session.measurement_count, 0)
        self.normalizer.normalize.assert_not_called()
        for _ in range(29):
            self.submit(face())
        self.assertEqual(self.session.measurement_count, 29)
        self.assertIsNone(self.session.reference)
        self.submit(face())
        self.assertEqual(self.session.measurement_count, 30)
        self.assertEqual(self.normalizer.normalize.call_count, 30)
        self.assertEqual(self.session.phase, Phase.WAITING)
        self.assertIsNotNone(self.session.reference)

    def test_repeated_and_older_timestamps_do_not_fill_warmup_or_measurement(self) -> None:
        self.session.start_stage()
        self.submit(face())
        repeated = LandmarkResult((face(0.2),), self.timestamp, 10)
        for _ in range(8):
            self.assertFalse(self.session.observe(repeated))
        self.assertFalse(self.session.observe(LandmarkResult((), 0, 10)))
        self.assertEqual(self.session.warmup_count, 1)
        for _ in range(4):
            self.submit(face())
        self.submit(face())
        repeated = LandmarkResult((face(0.2),), self.timestamp, 10)
        for _ in range(8):
            self.assertFalse(self.session.observe(repeated))
        self.assertEqual(self.session.measurement_count, 1)
        self.normalizer.normalize.assert_called_once()

    def test_waiting_and_cross_stage_snapshots_are_not_samples(self) -> None:
        self.submit(face())
        self.assertEqual(self.session.warmup_count, 0)
        self.session.start_stage()
        self.assertFalse(self.session.observe(LandmarkResult((face(),), self.timestamp, 10)))
        self.assertEqual(self.session.warmup_count, 0)
        for _ in range(35):
            self.submit(face())
        last = LandmarkResult((face(),), self.timestamp, 10)
        self.assertEqual(self.session.phase, Phase.WAITING)
        for _ in range(5):
            self.submit(face(0.1))
        self.assertEqual(self.session.reports, [])
        self.session.start_stage()
        self.assertFalse(self.session.observe(last))
        self.assertEqual(self.session.measurement_count, 0)
        self.assertEqual(self.session.warmup_count, 0)

    def test_missing_faces_count_as_warmup_but_not_measurements(self) -> None:
        self.session.start_stage()
        for _ in range(5):
            self.submit()
        self.assertEqual(self.session.warmup_count, 5)
        self.submit()
        self.assertEqual(self.session.measurement_count, 0)
        self.assertIn("No face", self.session.error)
        self.submit(face())
        self.assertEqual(self.session.measurement_count, 1)
        self.assertIsNone(self.session.error)

    def test_invalid_normalization_and_mismatched_counts_do_not_fill_slots(self) -> None:
        self.session.start_stage()
        for _ in range(5):
            self.submit()
        self.submit(FaceLandmarks((Landmark(0, 0, 0),)))
        self.assertIn("indices", self.session.error)
        self.assertEqual(self.session.measurement_count, 0)
        self.submit(face())
        self.submit(face(count=364))
        self.assertIn("counts must match", self.session.error)
        self.assertEqual(self.session.measurement_count, 1)
        for _ in range(29):
            self.submit(face())
        self.session.start_stage()
        for _ in range(5):
            self.submit()
        self.submit(face(count=364))
        self.assertIn("counts must match", self.session.error)
        self.assertEqual(self.session.measurement_count, 0)

    def test_nonfinite_coordinates_do_not_fill_measurement_slots(self) -> None:
        self.session.start_stage()
        for _ in range(5):
            self.submit()
        points = list(face().points)
        points[0] = Landmark(float("nan"), 0, 0)
        self.submit(FaceLandmarks(tuple(points)))
        self.assertIn("Nonfinite", self.session.error)
        self.assertEqual(self.session.measurement_count, 0)

    def test_stage_order_is_exact_and_every_stage_requires_explicit_start(self) -> None:
        expected = ["CENTER", "TRANSLATION LEFT", "TRANSLATION RIGHT", "TRANSLATION UP",
                    "TRANSLATION DOWN", "NEAR", "FAR", "ROLL LEFT", "ROLL RIGHT",
                    "YAW LEFT", "YAW RIGHT", "PITCH UP", "PITCH DOWN"]
        self.assertEqual([condition.name for condition in CONDITIONS], expected)
        self.assertEqual(STAGES[0].name, "REFERENCE")
        for stage in STAGES:
            self.assertEqual(self.session.condition, stage)
            self.assertEqual(self.session.phase, Phase.WAITING)
            self.complete_stage()
            self.submit(face())
        self.assertEqual(self.session.phase, Phase.DONE)
        self.assertEqual([report.condition.name for report in self.session.reports], expected)
        self.assertTrue(all(report.samples == 30 for report in self.session.reports))
        self.assertFalse(self.session.start_stage())
        self.assertIsNone(self.session.condition)

    def test_space_during_collection_does_not_reset_counts(self) -> None:
        self.session.start_stage()
        self.submit(face())
        self.assertFalse(self.session.start_stage())
        self.assertEqual(self.session.warmup_count, 1)


if __name__ == "__main__":
    unittest.main()
