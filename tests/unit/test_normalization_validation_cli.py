from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import Mock, patch
import unittest

from face_recognition.application.normalization_validation import (
    NormalizationValidation, Phase, ValidationConfig,
)
from face_recognition.config.settings import Settings
from face_recognition.domain.models.face_landmarks import FaceLandmarks, Landmark
from face_recognition.domain.models.landmark_result import DetectorSnapshot, LandmarkResult
from face_recognition.infrastructure.features.landmark_normalizer import LandmarkNormalizer
from face_recognition.presentation.cli import validate_normalization as cli


class GuidedCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.camera_factory = self.patch("OpenCVCamera")
        self.detector_factory = self.patch("MediaPipeFaceLandmarker")
        self.patch("FrameRenderer")
        self.gui = self.patch("cv2")
        self.keyboard = self.patch("terminal_key")
        self.camera = self.camera_factory.return_value
        self.detector = self.detector_factory.return_value
        self.camera.read.return_value = object()
        self.detector.snapshot.return_value = DetectorSnapshot()
        self.gui.waitKey.return_value = 255
        self.keyboard.return_value = "q"
        self.output = StringIO()
        redirect = redirect_stdout(self.output)
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)

    def patch(self, name: str) -> Mock:
        patcher = patch.object(cli, name)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def assert_closed(self) -> None:
        self.gui.destroyAllWindows.assert_called_once()
        self.camera.close.assert_called_once()
        self.detector.close.assert_called_once()

    def test_terminal_quit_is_nonblocking_and_closes_resources(self) -> None:
        cli.run_validation(Settings())
        self.assertIn("REFERENCE", self.output.getvalue())
        self.assertIn("Press SPACE when ready", self.output.getvalue())
        self.detector.submit.assert_called_once()
        self.assert_closed()

    def test_ctrl_c_cleans_up(self) -> None:
        self.keyboard.side_effect = KeyboardInterrupt
        cli.run_validation(Settings())
        self.assertIn("interrupted", self.output.getvalue())
        self.assert_closed()

    def test_camera_initialization_failure_closes_detector(self) -> None:
        self.camera_factory.side_effect = RuntimeError("camera unavailable")
        with self.assertRaisesRegex(RuntimeError, "camera unavailable"):
            cli.run_validation(Settings())
        self.detector.close.assert_called_once()

    def test_gui_failure_cleans_up(self) -> None:
        self.gui.imshow.side_effect = RuntimeError("GUI error")
        with self.assertRaisesRegex(RuntimeError, "GUI error"):
            cli.run_validation(Settings())
        self.assert_closed()

    def test_complete_guided_session_prints_reports_and_summary_without_devices(self) -> None:
        points = [Landmark(0.5, 0.5, 0) for _ in range(363)]
        points[33] = points[133] = Landmark(0.4, 0.4, 0)
        points[362] = points[263] = Landmark(0.6, 0.4, 0)
        sample = FaceLandmarks(tuple(points))
        timestamp = 0

        def snapshot() -> DetectorSnapshot:
            nonlocal timestamp
            timestamp += 1
            return DetectorSnapshot(LandmarkResult((sample,), timestamp, 10))

        self.detector.snapshot.side_effect = snapshot
        # Pressing SPACE during a collection must not start the next stage early.
        self.keyboard.return_value = " "
        config = ValidationConfig(1, 2)
        session = NormalizationValidation(LandmarkNormalizer(480, 270), 480, 270, config)
        with patch.object(cli, "NormalizationValidation", return_value=session):
            cli.run_validation(Settings(), config)
        self.assertEqual(session.phase, Phase.DONE)
        self.assertEqual(len(session.reports), 13)
        self.assertEqual(self.camera.read.call_count, 56)  # 14 × (SPACE + 1 warm-up + 2 samples)
        output = self.output.getvalue()
        self.assertIn("TEST 1/13 — CENTER", output)
        self.assertIn("TEST 13/13 — PITCH DOWN", output)
        self.assertIn("Warm-up: 1/1", output)
        self.assertIn("Samples: 2", output)
        self.assertIn("NORMALIZATION VALIDATION SUMMARY", output)
        self.assertIn("Raw median", output)
        self.assertIn("Norm median", output)
        self.assert_closed()


class TerminalKeyTests(unittest.TestCase):
    def test_console_poll_returns_immediately_when_no_key_is_available(self) -> None:
        console = Mock()
        console.kbhit.return_value = False
        with patch.object(cli.os, "name", "nt"), patch.dict("sys.modules", {"msvcrt": console}):
            self.assertIsNone(cli.terminal_key())
        console.getwch.assert_not_called()

    def test_console_space_needs_no_enter_and_control_c_exits(self) -> None:
        console = Mock()
        console.kbhit.return_value = True
        console.getwch.side_effect = [" ", "\x03"]
        with patch.object(cli.os, "name", "nt"), patch.dict("sys.modules", {"msvcrt": console}):
            self.assertEqual(cli.terminal_key(), " ")
            with self.assertRaises(KeyboardInterrupt):
                cli.terminal_key()


if __name__ == "__main__":
    unittest.main()
