import unittest

import numpy as np

from face_recognition.domain.models.face_landmarks import FaceLandmarks, Landmark
from face_recognition.presentation.visualization.frame_renderer import FrameRenderer


class RendererTests(unittest.TestCase):
    def test_renderer_draws_without_modifying_source_frame(self) -> None:
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        face = FaceLandmarks((Landmark(0.5, 0.5, 0.0),))
        rendered = FrameRenderer().render(frame, [face], 25.0)
        self.assertTrue(np.any(rendered))
        self.assertFalse(np.any(frame))
