"""Stable decision overlay; candidate identities are deliberately debug-only."""

from math import ceil, floor

import cv2
import numpy as np

from face_recognition.presentation.visualization.frame_renderer import FrameRenderer


def face_box(face, width, height, padding=.08):
    if face is None or width < 1 or height < 1:
        return None
    xy = np.array([(p.x, p.y) for p in face.points], dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or not len(xy) or not np.isfinite(xy).all():
        return None
    xy = np.clip(xy, 0., 1.) * (width - 1, height - 1)
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    extra = np.maximum(hi - lo, 1.) * padding
    return (max(0, floor(lo[0] - extra[0])), max(0, floor(lo[1] - extra[1])),
            min(width - 1, ceil(hi[0] + extra[0])), min(height - 1, ceil(hi[1] + extra[1])))


class LiveOpenSetRenderer(FrameRenderer):
    def __init__(self, debug=False):
        super().__init__("all")  # Enables per-completion DRM refresh, not mesh drawing.
        self.debug = debug
        self.view = self.live_metrics = None
        self.revision = 0

    def update(self, view, live_metrics):
        self.view, self.live_metrics, self.revision = view, live_metrics, view.revision

    def metric_lines(self, metrics, display):
        view, live = self.view, self.live_metrics
        if view is None or live is None:
            return ("NO FACE",)
        score = "pending" if view.last_score is None else f"{view.last_score:.2f}"
        latency = "pending" if live.latency_ms is None else f"{live.latency_ms:.1f} ms"
        lines = (view.label, f"Last score (completed window): {score}", f"Threshold: {view.threshold:.2f}",
                 f"Inference FPS: {live.inference_fps:.1f}", f"Latency (submission to V3): {latency}",
                 f"Window: {view.occupancy}/{view.window_size}", f"Camera FPS (reads): {metrics.capture_fps:.1f}")
        if self.debug:
            current_score = "pending" if view.frame_score is None else f"{view.frame_score:.2f}"
            lines += (f"Frame candidate: {view.frame_candidate or 'pending'} / score: {current_score}",
                      f"Temporal candidate: {view.temporal_candidate or 'pending'}",
                      f"Temporal decision: {view.temporal_decision or 'pending'}",
                      f"V3 face completions: {live.completed_face_inferences} / MP callbacks: {metrics.completed_inference_frames} / busy skips: {metrics.skipped_busy_frames}")
        return lines

    def _draw(self, output, faces, metrics, display):
        height, width = output.shape[:2]
        color = (0, 255, 0, 255) if output.shape[2] == 4 else (0, 255, 0)
        box = face_box(None if self.view is None else self.view.face, width, height)
        if box is not None:
            cv2.rectangle(output, box[:2], box[2:], color, 2)
        for row, text in enumerate(self.metric_lines(metrics, display)):
            cv2.putText(output, text, (10, 20 + row * 20), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
        hint = "q / ESC / Ctrl+C to quit" if display == "opencv" else "Ctrl+C to quit"
        cv2.putText(output, hint, (10, max(12, height - 10)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
