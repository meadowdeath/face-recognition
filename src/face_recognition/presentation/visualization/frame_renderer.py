"""OpenCV-only visualization for the landmark preview."""

from collections.abc import Sequence

import cv2
import numpy as np

from face_recognition.application.performance import PerformanceMetrics
from face_recognition.domain.models.face_landmarks import FaceLandmarks


class FrameRenderer:
    MODES = ("none", "selected", "all")

    def __init__(self, mode: str = "all", selected_indices: tuple[int, ...] = ()) -> None:
        if mode not in self.MODES:
            raise ValueError(f"Unknown renderer mode: {mode}")
        if any(index < 0 for index in selected_indices):
            raise ValueError("Landmark indices must be nonnegative")
        if mode == "selected" and not selected_indices:
            raise ValueError("Selected mode requires explicit landmark indices")
        self.mode = mode
        self.selected_indices = selected_indices

    def render(
        self,
        frame: np.ndarray,
        faces: Sequence[FaceLandmarks],
        metrics: PerformanceMetrics,
    ) -> np.ndarray:
        output = frame.copy()
        height, width = output.shape[:2]
        for face in (faces if self.mode != "none" else ()):
            indices = range(len(face.points)) if self.mode == "all" else self.selected_indices
            for index in indices:
                if index >= len(face.points):
                    continue
                point = face.points[index]
                x = round(point.x * (width - 1))
                y = round(point.y * (height - 1))
                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(output, (x, y), 1, (0, 255, 0), -1)
        latency = "pending" if metrics.result_latency_ms is None else f"{metrics.result_latency_ms:.1f} ms"
        lines = (
            f"Capture: {metrics.capture_fps:.1f} FPS  Display: {metrics.display_fps:.1f} FPS",
            f"Inference: {metrics.inference_fps:.1f} FPS  Mode: {self.mode}",
            f"Submitted: {metrics.submitted_inference_frames}  Completed: {metrics.completed_inference_frames}",
            f"Result latency: {latency}",
        )
        for row, text in enumerate(lines):
            cv2.putText(output, text, (10, 22 + row * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(output, "Press q to quit", (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return output
