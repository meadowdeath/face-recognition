import unittest

from face_recognition.application.recognize_face import LandmarkPreview
from face_recognition.domain.interfaces.camera import Camera
from face_recognition.domain.interfaces.landmark_detector import LandmarkDetector
from face_recognition.domain.models.face_landmarks import FaceLandmarks, Landmark


class PreviewTests(unittest.TestCase):
    def test_camera_and_detector_orchestration_without_hardware(self) -> None:
        frame = object()
        face = FaceLandmarks((Landmark(0.5, 0.5, 0.0),))

        class FakeCamera:
            closed = False
            calls = 0

            def read(self) -> object | None:
                self.calls += 1
                return frame if self.calls == 1 else None

            def close(self) -> None:
                self.closed = True

        class FakeDetector:
            closed = False
            seen = None

            def detect(self, captured: object) -> list[FaceLandmarks]:
                self.seen = captured
                return [face]

            def close(self) -> None:
                self.closed = True

        camera: Camera = FakeCamera()
        detector: LandmarkDetector = FakeDetector()
        preview = LandmarkPreview(camera, detector)
        result = preview.next_frame()
        self.assertIsNotNone(result)
        self.assertIs(result.frame, frame)
        self.assertEqual(result.faces, [face])
        self.assertIs(detector.seen, frame)
        self.assertIsNone(preview.next_frame())
        preview.close()
        self.assertTrue(camera.closed and detector.closed)
