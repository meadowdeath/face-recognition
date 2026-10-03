"""Display selection lives in presentation, outside capture and inference."""

from collections.abc import Callable
from time import perf_counter
from typing import Protocol

import cv2
import numpy as np

from face_recognition.application.performance import PerformanceMetrics
from face_recognition.application.recognize_face import LandmarkFrame
from face_recognition.presentation.visualization.frame_renderer import FrameRenderer


class PreviewDisplay(Protocol):
    def show(self, frame: LandmarkFrame, metrics: PerformanceMetrics) -> bool:
        """Present the current state; return False when the user requests exit."""
        ...

    def close(self) -> None: ...


class OverlayTarget(Protocol):
    """Optional preview capability, separate from the domain Camera protocol."""

    def start_drm_preview(self, width: int, height: int) -> None: ...

    def set_overlay(self, overlay: np.ndarray | None) -> None: ...

    def stop_preview(self) -> None: ...


class OpenCVDisplay:
    def __init__(self, renderer: FrameRenderer) -> None:
        self.renderer = renderer

    def show(self, frame: LandmarkFrame, metrics: PerformanceMetrics) -> bool:
        cv2.imshow("Face Landmarks", self.renderer.render(frame.frame, frame.faces, metrics))
        return cv2.waitKey(1) & 0xFF != ord("q")

    def close(self) -> None:
        cv2.destroyAllWindows()


class DRMDisplay:
    def __init__(
        self, target: OverlayTarget, renderer: FrameRenderer, width: int, height: int,
        interval_seconds: float = 1.0, clock: Callable[[], float] = perf_counter,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("Overlay metrics interval must be positive")
        self.target = target
        self.renderer = renderer
        self.width = width
        self.height = height
        self._interval = interval_seconds
        self._clock = clock
        self._last_metrics_refresh = clock()
        self._last_result_timestamp: int | None = None
        self._has_overlay = False
        self.overlay_updated = False
        self._closed = False
        target.start_drm_preview(width, height)

    def show(self, frame: LandmarkFrame, metrics: PerformanceMetrics) -> bool:
        self.overlay_updated = False
        now = self._clock()
        result = frame.detector_snapshot.result
        timestamp = None if result is None else result.timestamp_ms
        new_landmarks = (
            self.renderer.mode != "none" and timestamp is not None
            and timestamp != self._last_result_timestamp
        )
        metrics_due = now - self._last_metrics_refresh >= self._interval
        if self._has_overlay and not new_landmarks and not metrics_due:
            return True
        # Allocate/draw only when submitting an overlay. frame.faces contains
        # the latest completed landmarks, also used for metric-only refreshes.
        overlay = self.renderer.render_overlay(self.width, self.height, frame.faces, metrics)
        self.target.set_overlay(overlay)
        self.overlay_updated = True
        if not self._has_overlay or metrics_due:
            self._last_metrics_refresh = now
        self._has_overlay = True
        self._last_result_timestamp = timestamp
        return True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.target.set_overlay(None)
        finally:
            self.target.stop_preview()


class NoDisplay:
    """Console-only benchmark reporting; no frame or overlay rendering."""

    def __init__(
        self,
        renderer: FrameRenderer,
        interval_seconds: float,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self.renderer = renderer
        self._interval = interval_seconds
        self._clock = clock
        self._last_report = clock()

    def show(self, frame: LandmarkFrame, metrics: PerformanceMetrics) -> bool:
        now = self._clock()
        if now - self._last_report >= self._interval:
            print(" | ".join(self.renderer.metric_lines(metrics, "none")), flush=True)
            self._last_report = now
        return True

    def close(self) -> None:
        pass
