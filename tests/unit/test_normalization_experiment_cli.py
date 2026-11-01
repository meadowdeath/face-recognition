import unittest
from unittest.mock import Mock, patch

from face_recognition.application.normalization_experiment import NormalizationExperiment
from face_recognition.config.settings import Settings
from face_recognition.domain.models.face_landmarks import FaceLandmarks, Landmark
from face_recognition.domain.models.landmark_result import DetectorSnapshot, LandmarkResult
from face_recognition.infrastructure.features.landmark_normalizer import LandmarkNormalizer
from face_recognition.presentation.cli import normalize_landmarks as cli


class ExperimentCliTests(unittest.TestCase):
    def setUp(self) -> None:
        # Replace all device/GUI boundaries; no webcam, inference runtime, or windows.
        self.camera_factory = self.patch("OpenCVCamera")
        self.detector_factory = self.patch("MediaPipeFaceLandmarker")
        self.camera_renderer = self.patch("FrameRenderer")
        self.cloud_renderer = self.patch("NormalizedLandmarkRenderer")
        self.gui = self.patch("cv2")
        self.camera = self.camera_factory.return_value
        self.detector = self.detector_factory.return_value
        self.camera.read.return_value = object()
        self.detector.snapshot.return_value = DetectorSnapshot()
        self.gui.waitKey.return_value = ord("q")

    def patch(self, name: str) -> Mock:
        patcher = patch.object(cli, name)
        mocked = patcher.start()
        self.addCleanup(patcher.stop)
        return mocked

    def assert_closed(self) -> None:
        self.gui.destroyAllWindows.assert_called_once()
        self.detector.close.assert_called_once()
        self.camera.close.assert_called_once()

    def test_quit_reuses_preview_and_configured_camera_and_inference_dimensions(self) -> None:
        settings = Settings(inference_width=640, inference_height=360)
        cli.run_experiment(settings)
        self.camera_factory.assert_called_once_with(0, 848, 480, 30, orientation="normal")
        self.assertEqual(self.detector_factory.call_args.kwargs["inference_size"], (640, 360))
        self.detector.submit.assert_called_once_with(self.camera.read.return_value)
        self.assertEqual(self.gui.imshow.call_count, 2)
        self.assert_closed()

    def test_reference_key_uses_latest_completion_after_gui_event_pump(self) -> None:
        points = [Landmark(0.5, 0.5, 0) for _ in range(363)]
        points[33] = points[133] = Landmark(0.4, 0.4, 0)
        points[362] = points[263] = Landmark(0.6, 0.4, 0)
        first = FaceLandmarks(tuple(points))
        latest = FaceLandmarks(tuple(Landmark(p.x + 0.1, p.y, p.z) for p in points))
        snapshots = [DetectorSnapshot(LandmarkResult((face,), timestamp, 10))
                     for face, timestamp in ((first, 100), (latest, 101), (latest, 101))]
        self.detector.snapshot.side_effect = snapshots
        self.gui.waitKey.side_effect = [ord("r"), ord("q")]
        normalizer = Mock(wraps=LandmarkNormalizer(480, 270))
        experiment = NormalizationExperiment(normalizer, 480, 270)
        with patch.object(cli, "NormalizationExperiment", return_value=experiment):
            cli.run_experiment(Settings())
        self.assertIs(experiment.reference.raw, latest)
        self.assertEqual(experiment.observations, 2)
        self.assertEqual(normalizer.normalize.call_count, 2)
        self.assert_closed()

    def test_camera_initialization_failure_closes_detector(self) -> None:
        self.camera_factory.side_effect = RuntimeError("no camera")
        with self.assertRaisesRegex(RuntimeError, "no camera"):
            cli.run_experiment(Settings())
        self.detector.close.assert_called_once()
        self.gui.imshow.assert_not_called()

    def test_camera_read_failure_cleans_up(self) -> None:
        self.camera.read.return_value = None
        with self.assertRaisesRegex(RuntimeError, "stopped returning frames"):
            cli.run_experiment(Settings())
        self.assert_closed()

    def test_gui_failure_cleans_up(self) -> None:
        self.gui.imshow.side_effect = RuntimeError("display failed")
        with self.assertRaisesRegex(RuntimeError, "display failed"):
            cli.run_experiment(Settings())
        self.assert_closed()


if __name__ == "__main__":
    unittest.main()
