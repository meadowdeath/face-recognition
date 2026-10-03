import unittest

from face_recognition.application.performance import PerformanceTracker
from face_recognition.domain.models.landmark_result import DetectorSnapshot, LandmarkResult


class PerformanceTests(unittest.TestCase):
    def test_independent_rates_counts_and_latency(self) -> None:
        now = [0.0]
        tracker = PerformanceTracker(clock=lambda: now[0])
        for _ in range(30):
            tracker.record_capture()
        for _ in range(25):
            tracker.record_display()
        status = DetectorSnapshot(LandmarkResult((), 100, 42.5), 30, 15)
        now[0] = 1.0
        metrics = tracker.snapshot(status)
        self.assertEqual(metrics.capture_fps, 30)
        self.assertEqual(metrics.display_fps, 25)
        self.assertEqual(metrics.inference_fps, 15)
        self.assertEqual(metrics.submitted_inference_frames, 30)
        self.assertEqual(metrics.completed_inference_frames, 15)
        self.assertEqual(metrics.result_latency_ms, 42.5)
        now[0] = 1.5
        self.assertEqual(tracker.snapshot(status).inference_fps, 15)
        now[0] = 2.0
        idle = tracker.snapshot(status)
        self.assertEqual((idle.capture_fps, idle.display_fps, idle.inference_fps), (0, 0, 0))

    def test_startup_without_result_and_invalid_interval(self) -> None:
        tracker = PerformanceTracker(clock=lambda: 0.0)
        self.assertIsNone(tracker.snapshot(DetectorSnapshot()).result_latency_ms)
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
