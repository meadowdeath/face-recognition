import ast
import inspect
from pathlib import Path
import unittest

import numpy as np

from face_recognition.infrastructure.features.landmark_feature_extractor import LandmarkFeatureExtractor
from face_recognition.infrastructure.features.landmark_normalizer import LandmarkNormalizer
from face_recognition.infrastructure.features import pose_landmark_normalizer as geometry
from tests.pose_representation_fakes import encoded_face, face_geometry, rotation


class PoseLandmarkNormalizerTests(unittest.TestCase):
    def setUp(self):
        self.points = face_geometry()
        self.face = encoded_face(self.points)
        self.normalizer = geometry.PoseLandmarkNormalizer(640, 360)
        self.expected = self.array(self.normalizer.normalize(self.face))

    @staticmethod
    def array(face):
        return np.array([(p.x, p.y, p.z) for p in face.points])

    def canonical(self, points):
        return self.array(self.normalizer.normalize(encoded_face(points)))

    def assert_robust(self, points):
        np.testing.assert_allclose(self.canonical(points), self.expected, rtol=0, atol=1e-11)

    def test_deterministic_478_xyz_1434_finite_features_and_input_unchanged(self):
        before = self.face.points
        first = self.normalizer.normalize(self.face)
        second = self.normalizer.normalize(self.face)
        self.assertEqual(first, second)
        self.assertEqual(len(first.points), 478)
        values = LandmarkFeatureExtractor().extract(first).values
        self.assertEqual(len(values), 1434)
        self.assertTrue(np.isfinite(values).all())
        self.assertEqual(self.face.points, before)
        self.assertIsNot(first, self.face)

    def test_translation_robustness_including_depth(self):
        self.assert_robust(self.points + (1.2, -.7, 2.5))

    def test_uniform_scale_robustness(self):
        for scale in (.01, .5, 5.):
            with self.subTest(scale=scale): self.assert_robust(self.points * scale)

    def test_roll_left_and_right_robustness(self):
        for angle in (-.7, .7):
            with self.subTest(angle=angle): self.assert_robust(self.points @ rotation("z", angle).T)

    def compare_pose_with_baseline(self, axis, angle):
        baseline = LandmarkNormalizer(640, 360)
        changed = self.points @ rotation(axis, angle).T
        before = self.array(baseline.normalize(self.face))
        after = self.array(baseline.normalize(encoded_face(changed)))
        baseline_shift = np.linalg.norm(after - before)
        candidate_shift = np.linalg.norm(self.canonical(changed) - self.expected)
        self.assertGreater(baseline_shift, .01)
        self.assertLess(candidate_shift, baseline_shift * .01)
        self.assert_robust(changed)

    def test_yaw_left_has_smaller_displacement_than_baseline(self):
        self.compare_pose_with_baseline("y", -.5)

    def test_yaw_right_has_smaller_displacement_than_baseline(self):
        self.compare_pose_with_baseline("y", .5)

    def test_pitch_up_has_smaller_displacement_than_baseline(self):
        self.compare_pose_with_baseline("x", -.4)

    def test_pitch_down_has_smaller_displacement_than_baseline(self):
        self.compare_pose_with_baseline("x", .4)

    def test_combined_rigid_transform_and_aspect_ratios(self):
        matrix = rotation("z", .3) @ rotation("y", -.5) @ rotation("x", .2)
        points = 2.3 * (self.points @ matrix.T) + (.3, -.2, .8)
        for width, height in ((640, 360), (360, 640), (480, 480)):
            normalized = geometry.PoseLandmarkNormalizer(width, height).normalize(encoded_face(points, width, height))
            np.testing.assert_allclose(self.array(normalized), self.expected, rtol=0, atol=1e-11)

    def test_origin_scale_axes_and_handedness(self):
        first = self.expected[[33, 133]].mean(axis=0)
        second = self.expected[[362, 263]].mean(axis=0)
        np.testing.assert_allclose(first, (-.5, 0, 0), atol=1e-12)
        np.testing.assert_allclose(second, (.5, 0, 0), atol=1e-12)
        vertical = self.expected[[152, 148, 377]].mean(axis=0) - self.expected[[10, 109, 338]].mean(axis=0)
        self.assertGreater(vertical[1], 0.)
        self.assertAlmostEqual(vertical[2], 0., places=12)
        original_det = np.linalg.det(np.column_stack((self.points[1] - self.points[0], self.points[2] - self.points[0], self.points[3] - self.points[0])))
        canonical_det = np.linalg.det(np.column_stack((self.expected[1] - self.expected[0], self.expected[2] - self.expected[0], self.expected[3] - self.expected[0])))
        self.assertGreater(original_det * canonical_det, 0.)

    def test_no_condition_branch_or_identity_argument(self):
        self.assertEqual(tuple(inspect.signature(geometry.PoseLandmarkNormalizer.normalize).parameters), ("self", "landmarks"))
        tree = ast.parse(Path(geometry.__file__).read_text(encoding="utf-8"))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        self.assertFalse(names & {"condition", "person_id", "identity", "labels", "y_fit"})
        literals = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
        self.assertFalse(literals & {"yaw_left", "yaw_right", "pitch_up", "pitch_down", "roll_left", "roll_right"})

    def test_collapsed_or_tiny_3d_eyes_rejected(self):
        for distance in (0., 1e-10):
            points = self.points.copy()
            points[[33, 133]] = (.5, .4, 0.)
            points[[362, 263]] = (.5 + distance, .4, 0.)
            with self.assertRaises(ValueError): self.canonical(points)

    def test_collinear_nearly_collinear_or_collapsed_vertical_rejected(self):
        for height in (0., 1e-10):
            points = self.points.copy()
            points[[10, 109, 338]] = (.4, .35, .01)
            points[[152, 148, 377]] = (.6, .35 + height, .01)
            with self.assertRaises(ValueError): self.canonical(points)

    def test_nonfinite_input_and_overflowing_transformed_output_rejected(self):
        for value in (np.nan, np.inf, -np.inf, 1e308):
            points = self.points.copy(); points[100, 0] = value
            with self.subTest(value=value), self.assertRaises(ValueError): self.canonical(points)

    def test_wrong_count_or_dimensions_rejected(self):
        with self.assertRaises(ValueError): self.normalizer.normalize(encoded_face(self.points[:-1]))
        for width, height in ((0, 360), (640, -1), (True, 360), (640., 360)):
            with self.assertRaises(ValueError): geometry.PoseLandmarkNormalizer(width, height)

    def test_baseline_still_has_2d_interocular_scale_and_unrotated_depth(self):
        # Explicit old equations on this unrolled face. Baseline source is not
        # modified and retains projected 2D scale, unlike the pose candidate.
        baseline = self.array(LandmarkNormalizer(640, 360).normalize(self.face))
        origin = (self.points[[33, 133]].mean(axis=0) + self.points[[362, 263]].mean(axis=0)) / 2
        scale = np.linalg.norm((self.points[[362, 263]].mean(axis=0) - self.points[[33, 133]].mean(axis=0))[:2])
        np.testing.assert_allclose(baseline, (self.points - origin) / scale, atol=1e-12)
