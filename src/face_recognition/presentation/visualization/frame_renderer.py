"""Landmark drawing for BGR frames and transparent RGBA overlays."""

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
        self._draw(output, faces, metrics, "opencv")
        return output

    def render_overlay(
        self,
        width: int,
        height: int,
        faces: Sequence[FaceLandmarks],
        metrics: PerformanceMetrics,
    ) -> np.ndarray:
        # The native DRM preview supplies the camera image, not this canvas.
        output = np.zeros((height, width, 4), dtype=np.uint8)
        self._draw(output, faces, metrics, "drm")
        return output

    def metric_lines(self, metrics: PerformanceMetrics, display: str) -> tuple[str, ...]:
        latency = "pending" if metrics.result_latency_ms is None else f"{metrics.result_latency_ms:.1f} ms"
        age = "pending" if metrics.result_age_ms is None else f"{metrics.result_age_ms:.1f} ms"
        capture = f"Capture: {metrics.capture_fps:.1f} FPS"
        if display == "opencv":
            capture += f"  Display: {metrics.display_fps:.1f} FPS"
        elif display == "drm":
            capture += f"  Overlay updates: {metrics.overlay_update_fps:.1f}/s"
        rendering = "Rendering: disabled" if display == "none" else f"Mode: {self.mode}"
        return (
            capture,
            f"Inference: {metrics.inference_fps:.1f} FPS  {rendering}",
            f"Submitted: {metrics.submitted_inference_frames}  Completed: {metrics.completed_inference_frames}",
            f"Captured: {metrics.captured_frames}  Skipped Busy: {metrics.skipped_busy_frames}",
            f"Result latency: {latency} (callback)  Result age: {age}",
        )

    def _draw(
        self,
        output: np.ndarray,
        faces: Sequence[FaceLandmarks],
        metrics: PerformanceMetrics,
        display: str,
    ) -> None:
        height, width = output.shape[:2]
        color = (0, 255, 0, 255) if output.shape[2] == 4 else (0, 255, 0)
        for face in (faces if self.mode != "none" else ()):
            indices = range(len(face.points)) if self.mode == "all" else self.selected_indices
            for index in indices:
                if index >= len(face.points):
                    continue
                point = face.points[index]
                x = round(point.x * (width - 1))
                y = round(point.y * (height - 1))
                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(output, (x, y), 1, color, -1)
        lines = self.metric_lines(metrics, display)
        for row, text in enumerate(lines):
            cv2.putText(output, text, (10, 22 + row * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        hint = "Press q to quit" if display == "opencv" else "Ctrl+C to quit"
        cv2.putText(output, hint, (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
