import unittest

from face_recognition.application.performance import PerformanceTracker
from face_recognition.domain.models.landmark_result import DetectorSnapshot, LandmarkResult


class PerformanceTests(unittest.TestCase):
    def test_video_worker_counts_timings_and_completed_fps(self) -> None:
        now = [0.0]
        tracker = PerformanceTracker(clock=lambda: now[0])
        for _ in range(30):
            tracker.record_capture()
        status = DetectorSnapshot(
            inference_mode="video-worker", accepted_frames=30, submitted_frames=8,
            completed_frames=7, overwritten_pending_frames=21,
            preprocessing_ms=2.0, sync_inference_ms=100.0, total_worker_latency_ms=102.0,
            result_age_ms=120.0,
        )
        now[0] = 1.0
        metrics = tracker.snapshot(status)
        self.assertEqual(metrics.inference_mode, "video-worker")
        self.assertEqual((metrics.captured_frames, metrics.accepted_frames,
                          metrics.submitted_inference_frames, metrics.completed_inference_frames,
                          metrics.overwritten_pending_frames), (30, 30, 8, 7, 21))
        self.assertEqual(metrics.inference_fps, 7.0)
        self.assertEqual((metrics.preprocessing_ms, metrics.sync_inference_ms,
                          metrics.total_worker_latency_ms, metrics.result_age_ms), (2.0, 100.0, 102.0, 120.0))
        self.assertEqual(metrics.skipped_busy_frames, 0)
        self.assertIsNone(metrics.total_callback_latency_ms)
    def test_latest_diagnostic_timings_are_forwarded_without_changing_counters(self) -> None:
        status = DetectorSnapshot(
            result=LandmarkResult((), 100, 115.0), submitted_frames=2, completed_frames=1,
            skipped_busy_frames=3, result_age_ms=160.0, preprocessing_ms=3.0,
            dispatch_call_ms=0.5, async_result_ms=111.5, total_callback_latency_ms=115.0,
        )
        tracker = PerformanceTracker(clock=lambda: 0.0)
        metrics = tracker.snapshot(status)
        self.assertEqual((metrics.preprocessing_ms, metrics.dispatch_call_ms,
                          metrics.async_result_ms, metrics.total_callback_latency_ms), (3.0, 0.5, 111.5, 115.0))
        self.assertEqual(metrics.result_latency_ms, 115.0)
        self.assertEqual(metrics.result_age_ms, 160.0)
        self.assertEqual((metrics.submitted_inference_frames, metrics.completed_inference_frames,
                          metrics.skipped_busy_frames), (2, 1, 3))
        pending = tracker.snapshot(DetectorSnapshot())
        self.assertIsNone(pending.preprocessing_ms)
        self.assertIsNone(pending.dispatch_call_ms)
        self.assertIsNone(pending.async_result_ms)
        self.assertIsNone(pending.total_callback_latency_ms)
    def test_independent_rates_counts_and_latency(self) -> None:
        now = [0.0]
        tracker = PerformanceTracker(clock=lambda: now[0])
        for _ in range(30):
            tracker.record_capture()
        for _ in range(25):
            tracker.record_display()
        status = DetectorSnapshot(LandmarkResult((), 100, 42.5), 16, 15, 14, 900.0)
        now[0] = 1.0
        metrics = tracker.snapshot(status)
        self.assertEqual(metrics.capture_fps, 30)
        self.assertEqual(metrics.display_fps, 25)
        self.assertEqual(metrics.inference_fps, 15)
        self.assertEqual(metrics.captured_frames, 30)
        self.assertEqual(metrics.submitted_inference_frames, 16)
        self.assertEqual(metrics.completed_inference_frames, 15)
        self.assertEqual(metrics.skipped_busy_frames, 14)
        self.assertEqual(metrics.result_latency_ms, 42.5)
        self.assertEqual(metrics.result_age_ms, 900.0)
        now[0] = 1.5
        self.assertEqual(tracker.snapshot(status).inference_fps, 15)
        now[0] = 2.0
        idle = tracker.snapshot(status)
        self.assertEqual((idle.capture_fps, idle.display_fps, idle.inference_fps), (0, 0, 0))

    def test_startup_without_result_and_invalid_interval(self) -> None:
        tracker = PerformanceTracker(clock=lambda: 0.0)
        self.assertIsNone(tracker.snapshot(DetectorSnapshot()).result_latency_ms)
        self.assertIsNone(tracker.snapshot(DetectorSnapshot()).result_age_ms)
        self.assertEqual(tracker.snapshot(DetectorSnapshot()).skipped_busy_frames, 0)
        with self.assertRaises(ValueError):
            PerformanceTracker(interval_seconds=0)

    def test_overlay_updates_are_independent_of_opencv_display_iterations(self) -> None:
        now = [0.0]
        tracker = PerformanceTracker(clock=lambda: now[0])
        for _ in range(20):
            tracker.record_overlay_update()
        now[0] = 1.0
        metrics = tracker.snapshot(DetectorSnapshot())
        self.assertEqual(metrics.overlay_update_fps, 20)
        self.assertEqual(metrics.overlay_updates, 20)
        self.assertEqual(metrics.display_fps, 0)
        self.assertEqual(metrics.displayed_frames, 0)
