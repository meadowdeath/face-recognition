from dataclasses import asdict
import json
from math import hypot
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

from face_recognition.application.evaluate_normalization import (
    NormalizationEvaluationError, NormalizationQualityEvaluator, POSE_CONDITIONS, TARGETED_CONDITIONS,
)
from face_recognition.domain.models.dataset import CONDITIONS
from face_recognition.domain.models.processed_sample import BASELINE_REPRESENTATION, NORMALIZED_REPRESENTATION
from face_recognition.infrastructure.features.landmark_normalizer import LandmarkNormalizer
from face_recognition.infrastructure.features.normalization_reference import center_interocular_distance
from face_recognition.infrastructure.persistence.normalized_dataset_reader import NormalizedDatasetReader
from face_recognition.infrastructure.persistence.raw_dataset_reader import RawDatasetReader
from tests.normalization_quality_fakes import process_quality_fixture, snapshot, write_quality_session


class NormalizationQualityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory(prefix="synthetic_normalization_quality_")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.raw = self.directory / "raw"
        self.raw.mkdir()
        self.processed = self.directory / "processed"
        self.evaluator = NormalizationQualityEvaluator(center_interocular_distance)

    def evaluate(self):
        return self.evaluator.evaluate(RawDatasetReader(self.raw), NormalizedDatasetReader(self.processed))

    @staticmethod
    def row(report, condition):
        return next(row for row in report.conditions if row.condition == condition)

    def test_center_reference_is_a_mean_not_the_first_frame(self) -> None:
        write_quality_session(self.raw, center_offsets=(0.02, 0.06),
                              transforms={"translation_left": {"translation": (0.04, 0, 0)}})
        process_quality_fixture(self.raw, self.processed)
        report = self.evaluate()
        self.assertAlmostEqual(self.row(report, "translation_left").raw.mean, 0, delta=1e-12)
        self.assertAlmostEqual(self.row(report, "center").raw.mean, 0.02, delta=1e-12)

    def test_translation_errors_statistics_and_comparable_reduction(self) -> None:
        write_quality_session(self.raw, transforms={"translation_left": (
            {"translation": (0.1, 0, 0)}, {"translation": (0.3, 0, 0)},
        )})
        process_quality_fixture(self.raw, self.processed)
        report = self.evaluate()
        row = self.row(report, "translation_left")
        distance = hypot(0.2, 0.1)
        self.assertAlmostEqual(row.raw.mean, 0.2, delta=1e-12)
        self.assertAlmostEqual(row.raw.std, 0.1, delta=1e-12)
        self.assertAlmostEqual(row.raw.median, 0.2, delta=1e-12)
        self.assertAlmostEqual(row.comparable_raw.mean, 0.2 / distance, delta=1e-12)
        self.assertAlmostEqual(row.normalized.mean, 0, delta=1e-12)
        self.assertAlmostEqual(row.absolute_reduction, row.comparable_raw.mean - row.normalized.mean)
        self.assertAlmostEqual(row.percentage_reduction, 100, delta=1e-9)
        self.assertEqual(row.sample_pairs, 2)
        self.assertEqual(report.targeted_invariance.sample_pairs, len(TARGETED_CONDITIONS) * 2)
        self.assertAlmostEqual(report.targeted_invariance.raw.mean, 0.025, delta=1e-12)
        self.assertAlmostEqual(report.targeted_invariance.raw.std, 0.075, delta=1e-12)

    def test_moderate_scale_and_roll_have_lower_normalized_errors(self) -> None:
        write_quality_session(self.raw, transforms={"near": {"scale": 1.2}, "far": {"scale": 0.8},
                                                   "roll_left": {"roll": -0.3}, "roll_right": {"roll": 0.3}})
        process_quality_fixture(self.raw, self.processed)
        report = self.evaluate()
        for condition in ("near", "far", "roll_left", "roll_right"):
            with self.subTest(condition=condition):
                row = self.row(report, condition)
                self.assertGreater(row.comparable_raw.mean, 0.01)
                self.assertLess(row.normalized.mean, 1e-12)

    def test_yaw_and_pitch_are_separate_and_not_assumed_invariant(self) -> None:
        write_quality_session(self.raw, transforms={"yaw_left": {"yaw": -0.4}, "yaw_right": {"yaw": 0.4},
                                                   "pitch_up": {"pitch": 0.4}, "pitch_down": {"pitch": -0.4}})
        process_quality_fixture(self.raw, self.processed)
        report = self.evaluate()
        self.assertFalse(set(POSE_CONDITIONS) & set(TARGETED_CONDITIONS))
        for condition in POSE_CONDITIONS:
            self.assertGreater(self.row(report, condition).normalized.mean, 0.01)
        self.assertEqual(report.pose.sample_pairs, 8)
        self.assertLess(report.targeted_invariance.normalized.mean, 1e-12)

    def test_rectangular_raw_y_geometry_is_corrected_exactly_once(self) -> None:
        write_quality_session(self.raw, transforms={"translation_up": {"translation": (0, 0.04, 0)}},
                              width=480, height=270)
        process_quality_fixture(self.raw, self.processed)
        row = self.row(self.evaluate(), "translation_up")
        self.assertAlmostEqual(row.raw.mean, 0.04, delta=1e-12)
        self.assertLess(row.normalized.mean, 1e-12)
        self.assertNotAlmostEqual(row.raw.mean, 0.04 * 480 / 270)

    def test_person_session_references_stay_separate_with_two_and_three_sessions(self) -> None:
        for person, count in (("subject_a", 2), ("subject_b", 3)):
            for index in range(count):
                write_quality_session(self.raw, person, f"visit_{index}", identity_shape=0.1 * (index + count))
        process_quality_fixture(self.raw, self.processed)
        report = self.evaluate()
        self.assertEqual((report.participants, report.sessions, report.condition_files, report.evaluated_sample_pairs),
                         (2, 5, 65, 130))
        self.assertTrue(all(row.raw.mean < 1e-12 and row.normalized.mean < 1e-12 for row in report.conditions))
        self.assertEqual(report.targeted_invariance.sample_pairs, 80)
        self.assertEqual(report.pose.sample_pairs, 40)

    def test_arbitrary_counts_and_dimensions_are_supported(self) -> None:
        for index, (count, width, height) in enumerate(((363, 480, 270), (478, 270, 480), (500, 640, 480))):
            write_quality_session(self.raw, session=f"visit_{index}", count=count, width=width, height=height)
        process_quality_fixture(self.raw, self.processed)
        self.assertEqual(self.evaluate().evaluated_sample_pairs, 78)

    def test_aggregation_pools_samples_instead_of_averaging_session_means(self) -> None:
        write_quality_session(self.raw, session="visit_a", center_offsets=(0, 0),
                              transforms={"translation_left": {"translation": (0.1, 0, 0)}})
        write_quality_session(self.raw, session="visit_b", center_offsets=(0, 0, 0),
                              transforms={"translation_left": {"translation": (0.3, 0, 0)}})
        process_quality_fixture(self.raw, self.processed)
        report = self.evaluate()
        self.assertEqual(self.row(report, "translation_left").sample_pairs, 5)
        self.assertAlmostEqual(self.row(report, "translation_left").raw.mean, 0.22, delta=1e-12)
        self.assertEqual(report.targeted_invariance.sample_pairs, 40)
        self.assertAlmostEqual(report.targeted_invariance.raw.mean, 0.0275, delta=1e-12)

    def test_degenerate_center_geometry_fails_without_values_in_error(self) -> None:
        directory = write_quality_session(self.raw)
        process_quality_fixture(self.raw, self.processed)
        path = directory / "center.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            data = json.loads(line)
            for reference in (33, 133, 362, 263):
                data["landmarks"][reference] = [0, 0, 0]
            lines[index] = json.dumps(data)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(NormalizationEvaluationError, "invalid_center_reference") as caught:
            self.evaluate()
        self.assertNotIn(lines[0], str(caught.exception))

    def test_stored_normalized_values_are_not_normalized_again_and_sources_are_unchanged(self) -> None:
        write_quality_session(self.raw)
        process_quality_fixture(self.raw, self.processed)
        before_raw, before_processed = snapshot(self.raw), snapshot(self.processed)
        with patch.object(LandmarkNormalizer, "normalize", side_effect=AssertionError("Do not normalize stored data")):
            first, second = self.evaluate(), self.evaluate()
        self.assertEqual(asdict(first), asdict(second))
        self.assertTrue(snapshot(self.raw) == before_raw, "Evaluation must never rewrite raw data")
        self.assertTrue(snapshot(self.processed) == before_processed, "Evaluation must never rewrite processed data")

    def test_missing_center_condition_fails(self) -> None:
        write_quality_session(self.raw, conditions=())
        process_quality_fixture(self.raw, self.processed)
        with self.assertRaisesRegex(NormalizationEvaluationError, "missing_center_condition"):
            self.evaluate()

    def test_zero_baseline_percentage_is_undefined_and_negative_reductions_are_retained(self) -> None:
        write_quality_session(self.raw)
        process_quality_fixture(self.raw, self.processed)
        report = self.evaluate()
        self.assertIsNone(self.row(report, "translation_left").percentage_reduction)
        path = self.processed / NORMALIZED_REPRESENTATION / "synthetic_subject/synthetic_visit/translation_left.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            data = json.loads(line)
            for point in data["normalized_landmarks"]:
                point[0] += 1
            lines[index] = json.dumps(data)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        row = self.row(self.evaluate(), "translation_left")
        self.assertLess(row.absolute_reduction, 0)
        self.assertIsNone(row.percentage_reduction)
        raw_path = self.raw / "synthetic_subject/synthetic_visit/translation_left.jsonl"
        lines = raw_path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            data = json.loads(line)
            for point in data["landmarks"]:
                point[0] += 0.01
            lines[index] = json.dumps(data)
        raw_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.assertLess(self.row(self.evaluate(), "translation_left").percentage_reduction, 0)

    def test_features_are_not_required(self) -> None:
        write_quality_session(self.raw)
        process_quality_fixture(self.raw, self.processed)
        features = self.processed / BASELINE_REPRESENTATION
        destination = self.directory / "unused_feature_representation"
        self.assertTrue(features.resolve().is_relative_to(self.directory.resolve()))
        self.assertTrue(destination.resolve().is_relative_to(self.directory.resolve()))
        features.rename(destination)
        self.assertFalse(features.exists())
        self.assertEqual(self.evaluate().evaluated_sample_pairs, 26)

    def test_raw_processed_metadata_mismatch_and_manifest_counts_fail(self) -> None:
        write_quality_session(self.raw)
        process_quality_fixture(self.raw, self.processed)
        path = self.processed / NORMALIZED_REPRESENTATION / "synthetic_subject/synthetic_visit/translation_left.jsonl"
        baseline = path.read_text(encoding="utf-8")
        for changes in ({"person_id": "another_subject"}, {"session_id": "another_visit"},
                        {"condition": "translation_right"}, {"inference_width": 640},
                        {"detector_timestamp_ms": 987654}, {"sample_index": 1}):
            lines = baseline.splitlines()
            data = json.loads(lines[0])
            data.update(changes)
            lines[0] = json.dumps(data)
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.subTest(field=next(iter(changes))), self.assertRaises(NormalizationEvaluationError):
                self.evaluate()
        path.write_text(baseline, encoding="utf-8")
        header = self.processed / NORMALIZED_REPRESENTATION / "processing_manifest.json"
        data = json.loads(header.read_text(encoding="utf-8"))
        data["total_samples_processed"] += 1
        header.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(NormalizationEvaluationError, "sample_count_mismatch"):
            self.evaluate()
