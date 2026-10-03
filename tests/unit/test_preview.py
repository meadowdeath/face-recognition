import unittest

from face_recognition.application.recognize_face import LandmarkPreview
from face_recognition.domain.models.face_landmarks import FaceLandmarks, Landmark
from face_recognition.domain.models.landmark_result import DetectorSnapshot, LandmarkResult


class FakeCamera:
    def __init__(self, frames: list[object]) -> None:
        self.frames = iter(frames)
        self.closed = False

    def read(self) -> object | None:
        return next(self.frames, None)

    def close(self) -> None:
        self.closed = True


class FakeDetector:
    def __init__(self) -> None:
        self.submissions: list[object] = []
        self.latest: LandmarkResult | None = None
        self.completed = 0
        self.closed = False

    def submit(self, frame: object) -> None:
        self.submissions.append(frame)

    def snapshot(self) -> DetectorSnapshot:
        return DetectorSnapshot(self.latest, len(self.submissions), self.completed)

    def close(self) -> None:
        self.closed = True


class PreviewTests(unittest.TestCase):
    def test_capture_continues_without_an_inference_result(self) -> None:
        frames = [object() for _ in range(10)]
        camera = FakeCamera(frames)
        detector = FakeDetector()
        preview = LandmarkPreview(camera, detector)
        for frame in frames:
            result = preview.next_frame()
            self.assertIsNotNone(result)
            self.assertIs(result.frame, frame)
            self.assertEqual(result.faces, ())
        self.assertEqual(detector.submissions, frames)
        self.assertIsNone(preview.next_frame())
        metrics = preview.performance.snapshot(detector.snapshot())
        self.assertEqual(metrics.captured_frames, 10)
        self.assertEqual(metrics.completed_inference_frames, 0)
        preview.close()
        self.assertTrue(camera.closed and detector.closed)

    def test_latest_result_is_reused_and_empty_completion_clears_faces(self) -> None:
        camera = FakeCamera([object(), object(), object()])
        detector = FakeDetector()
        face = FaceLandmarks((Landmark(0.5, 0.5, 0.0),))
        detector.latest = LandmarkResult((face,), 100, 20.0)
        detector.completed = 1
        preview = LandmarkPreview(camera, detector)
        self.assertEqual(preview.next_frame().faces, (face,))
        self.assertEqual(preview.next_frame().faces, (face,))
        detector.latest = LandmarkResult((), 200, 15.0)
        detector.completed = 2
        self.assertEqual(preview.next_frame().faces, ())

    def test_camera_is_closed_even_if_detector_cleanup_fails(self) -> None:
        class FailingDetector(FakeDetector):
            def close(self) -> None:
                raise RuntimeError("cleanup failed")

        camera = FakeCamera([])
        preview = LandmarkPreview(camera, FailingDetector())
        with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
            preview.close()
        self.assertTrue(camera.closed)
