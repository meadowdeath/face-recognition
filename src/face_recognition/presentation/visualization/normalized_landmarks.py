"""Fixed presentation mapping for an eye-relative landmark cloud, not a face image."""

from math import isfinite

import cv2
import numpy as np

from face_recognition.application.normalization_experiment import NormalizationExperiment
from face_recognition.domain.models.face_landmarks import FaceLandmarks


class NormalizedLandmarkRenderer:
    SIZE = 640
    MIN_COORDINATE = -2.0
    MAX_COORDINATE = 2.0

    def canvas_point(self, x: float, y: float) -> tuple[int, int] | None:
        """Map [-2,2] to [0,639] on both axes; positive y points down."""
        if not (isfinite(x) and isfinite(y)):
            return None
        if not (self.MIN_COORDINATE <= x <= self.MAX_COORDINATE
                and self.MIN_COORDINATE <= y <= self.MAX_COORDINATE):
            return None
        scale = (self.SIZE - 1) / (self.MAX_COORDINATE - self.MIN_COORDINATE)
        return round((x - self.MIN_COORDINATE) * scale), round((y - self.MIN_COORDINATE) * scale)

    def _draw_cloud(self, canvas: np.ndarray, face: FaceLandmarks | None, color: tuple[int, int, int]) -> None:
        if face is None:
            return
        for point in face.points:
            pixel = self.canvas_point(point.x, point.y)
            if pixel is not None:
                cv2.circle(canvas, pixel, 1, color, -1)

    def render(self, experiment: NormalizationExperiment) -> np.ndarray:
        canvas = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)
        origin = self.canvas_point(0.0, 0.0)
        cv2.line(canvas, (0, origin[1]), (self.SIZE - 1, origin[1]), (50, 50, 50), 1)
        cv2.line(canvas, (origin[0], 0), (origin[0], self.SIZE - 1), (50, 50, 50), 1)
        if experiment.reference is not None:
            self._draw_cloud(canvas, experiment.reference.normalized, (130, 130, 130))
        self._draw_cloud(canvas, experiment.current_normalized, (0, 255, 0))
        for x in (-0.5, 0.5):
            cv2.circle(canvas, self.canvas_point(x, 0.0), 5, (255, 200, 0), 1)
        captured = experiment.reference is not None
        raw = "unavailable" if experiment.raw_rmse is None else f"{experiment.raw_rmse:.6f}"
        normalized = "unavailable" if experiment.normalized_rmse is None else f"{experiment.normalized_rmse:.6f}"
        lines = [
            "Identity-normalized cloud (green); reference (gray)",
            "Fixed X/Y: [-2, 2]; eye centers: +/-0.5, 0",
            f"Reference: {'captured' if captured else 'not captured'}",
        ]
        if captured:
            lines.extend((f"Raw RMSE: {raw} (image-width units)",
                          f"Normalized RMSE: {normalized} (interocular units)"))
        if experiment.error is not None:
            lines.append(f"Comparison unavailable: {experiment.error}")
        elif experiment.current_raw is None:
            lines.append("No completed face available")
        for row, line in enumerate(lines):
            cv2.putText(canvas, line, (10, 22 + row * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        cv2.putText(canvas, "r = capture/replace reference    q = quit", (10, self.SIZE - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        return canvas
