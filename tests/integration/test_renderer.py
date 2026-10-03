import unittest
from unittest.mock import patch

import numpy as np

from face_recognition.application.performance import PerformanceTracker
from face_recognition.domain.models.face_landmarks import FaceLandmarks, Landmark
from face_recognition.domain.models.landmark_result import DetectorSnapshot
from face_recognition.presentation.visualization.frame_renderer import FrameRenderer


class RendererTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = np.zeros((160, 200, 3), dtype=np.uint8)
        self.face = FaceLandmarks(tuple(Landmark(x, 0.7, 0.0) for x in (0.2, 0.5, 0.8)))
        self.metrics = PerformanceTracker().snapshot(DetectorSnapshot())

    def test_renderer_draws_without_modifying_source_frame(self) -> None:
        rendered = FrameRenderer().render(self.frame, [self.face], self.metrics)
        self.assertTrue(np.any(rendered))
        self.assertFalse(np.any(self.frame))

    def test_modes_draw_only_requested_points_and_keep_metrics(self) -> None:
        for mode, indices, expected in (("none", (), 0), ("selected", (1, 99), 1), ("all", (), 3)):
            with self.subTest(mode=mode), patch("cv2.circle") as circle, patch("cv2.putText") as text:
                FrameRenderer(mode, indices).render(self.frame, [self.face], self.metrics)
                self.assertEqual(circle.call_count, expected)
                labels = [call.args[1] for call in text.call_args_list]
                self.assertTrue(any("Capture:" in label and "Display:" in label for label in labels))
                self.assertTrue(any("Inference:" in label for label in labels))
                self.assertTrue(any("Submitted:" in label and "Completed:" in label for label in labels))

    def test_invalid_renderer_configuration(self) -> None:
        for mode, indices in (("invalid", ()), ("selected", ()), ("selected", (-1,))):
            with self.assertRaises(ValueError):
                FrameRenderer(mode, indices)

    def test_rgba_overlay_is_transparent_except_for_requested_annotations(self) -> None:
        for mode, indices, visible_points in (("none", (), ()), ("selected", (1,), (1,)),
                                             ("all", (), (0, 1, 2))):
            with self.subTest(mode=mode):
                overlay = FrameRenderer(mode, indices).render_overlay(200, 160, [self.face], self.metrics)
                self.assertEqual(overlay.shape, (160, 200, 4))
                self.assertEqual(overlay.dtype, np.uint8)
                self.assertEqual(tuple(overlay[100, 0]), (0, 0, 0, 0))
                for index, point in enumerate(self.face.points):
                    alpha = overlay[round(point.y * 159), round(point.x * 199), 3]
                    self.assertEqual(alpha, 255 if index in visible_points else 0)
                self.assertTrue(np.any(overlay[:, :, 3] == 255))  # Metrics remain visible.

    def test_drm_metrics_name_overlay_updates_and_do_not_claim_display_fps(self) -> None:
        lines = FrameRenderer().metric_lines(self.metrics, "drm")
        self.assertTrue(any("Overlay updates:" in line for line in lines))
        self.assertFalse(any("Display:" in line for line in lines))
        self.assertTrue(any("Result latency:" in line for line in lines))

    def test_normalized_inference_points_scale_to_full_display_overlay(self) -> None:
        face = FaceLandmarks((Landmark(0.0, 0.0, 0.0), Landmark(0.5, 0.5, 0.0),
                              Landmark(1.0, 1.0, 0.0)))
        with patch("cv2.circle") as circle:
            overlay = FrameRenderer().render_overlay(848, 480, [face], self.metrics)
        self.assertEqual(overlay.shape, (480, 848, 4))
        self.assertEqual([call.args[1] for call in circle.call_args_list],
                         [(0, 0), (424, 240), (847, 479)])
