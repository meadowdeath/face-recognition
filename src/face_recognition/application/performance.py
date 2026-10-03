"""Main-loop metrics with fixed storage and independently counted stages."""

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

from face_recognition.domain.models.landmark_result import DetectorSnapshot


@dataclass(frozen=True)
class PerformanceMetrics:
    capture_fps: float
    display_fps: float
    inference_fps: float
    captured_frames: int
    displayed_frames: int
    submitted_inference_frames: int
    completed_inference_frames: int
    result_latency_ms: float | None
    overlay_update_fps: float = 0.0
    overlay_updates: int = 0
    skipped_busy_frames: int = 0
    result_age_ms: float | None = None
    preprocessing_ms: float | None = None
    dispatch_call_ms: float | None = None
    async_result_ms: float | None = None
    total_callback_latency_ms: float | None = None
    inference_mode: str = "live-stream"
    accepted_frames: int = 0
    overwritten_pending_frames: int = 0
    sync_inference_ms: float | None = None
    total_worker_latency_ms: float | None = None


class PerformanceTracker:
    def __init__(
        self,
        interval_seconds: float = 1.0,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("Metrics interval must be positive")
        self._clock = clock
        self._interval = interval_seconds
        self._sample_time = clock()
        self._captured = 0
        self._displayed = 0
        self._overlay_updates = 0
        self._sample_counts = (0, 0, 0, 0)
        self._rates = (0.0, 0.0, 0.0, 0.0)

    def record_capture(self) -> None:
        self._captured += 1

    def record_display(self) -> None:
        self._displayed += 1

    def record_overlay_update(self) -> None:
        self._overlay_updates += 1

    def snapshot(self, detector: DetectorSnapshot) -> PerformanceMetrics:
        now = self._clock()
        elapsed = now - self._sample_time
        counts = (self._captured, self._displayed, detector.completed_frames, self._overlay_updates)
        if elapsed >= self._interval:
            self._rates = tuple(
                (count - previous) / elapsed
                for count, previous in zip(counts, self._sample_counts)
            )
            self._sample_counts = counts
            self._sample_time = now
        return PerformanceMetrics(
            capture_fps=self._rates[0],
            display_fps=self._rates[1],
            inference_fps=self._rates[2],
            captured_frames=self._captured,
            displayed_frames=self._displayed,
            submitted_inference_frames=detector.submitted_frames,
            completed_inference_frames=detector.completed_frames,
            result_latency_ms=None if detector.result is None else detector.result.latency_ms,
            overlay_update_fps=self._rates[3],
            overlay_updates=self._overlay_updates,
            skipped_busy_frames=detector.skipped_busy_frames,
            result_age_ms=detector.result_age_ms,
            preprocessing_ms=detector.preprocessing_ms,
            dispatch_call_ms=detector.dispatch_call_ms,
            async_result_ms=detector.async_result_ms,
            total_callback_latency_ms=detector.total_callback_latency_ms,
            inference_mode=detector.inference_mode,
            accepted_frames=detector.accepted_frames,
            overwritten_pending_frames=detector.overwritten_pending_frames,
            sync_inference_ms=detector.sync_inference_ms,
            total_worker_latency_ms=detector.total_worker_latency_ms,
        )
