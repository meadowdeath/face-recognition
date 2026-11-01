from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from unittest.mock import Mock, patch
import unittest

from face_recognition.config.settings import Settings
from face_recognition.domain.models.dataset import CONDITIONS, CaptureProtocol
from face_recognition.domain.models.landmark_result import DetectorSnapshot, LandmarkResult
from face_recognition.presentation.cli import capture as cli
from tests.dataset_fakes import FakeDatasetRepository, raw_face, session_config


class CaptureCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = session_config()
        self.repository = FakeDatasetRepository(self.config)
        self.camera_factory = self.patch("OpenCVCamera")
        self.pi_factory = self.patch("Picamera2Camera")
        self.detector_factory = self.patch("MediaPipeFaceLandmarker")
        self.opencv_display = self.patch("OpenCVDisplay")
        self.drm_display = self.patch("DRMDisplay")
        self.controls_factory = self.patch("TerminalControls")
        self.controls = self.controls_factory.return_value.__enter__.return_value
        self.controls.poll.return_value = "q"
        self.camera = self.camera_factory.return_value
        self.camera.read.return_value = object()
        self.pi_factory.return_value = self.camera
        self.detector = self.detector_factory.return_value
        self.detector.snapshot.return_value = DetectorSnapshot()
        self.opencv_display.return_value.show.return_value = True
        self.drm_display.return_value.show.return_value = True
        self.output = StringIO()
        redirect = redirect_stdout(self.output)
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)

    def patch(self, name: str) -> Mock:
        patcher = patch.object(cli, name)
        mocked = patcher.start()
        self.addCleanup(patcher.stop)
        return mocked

    def assert_closed(self) -> None:
        self.detector.close.assert_called_once()
        self.camera.close.assert_called_once()
        self.controls_factory.return_value.__exit__.assert_called_once()

    def test_windows_q_uses_existing_pipeline_without_picamera2(self) -> None:
        with patch.dict("sys.modules", {"picamera2": None, "libcamera": None}):
            cli.run_capture(Settings(), self.config, self.repository, renderer=Mock())
        self.camera_factory.assert_called_once_with(0, 848, 480, 30, orientation="normal")
        self.pi_factory.assert_not_called()
        self.detector.submit.assert_called_once_with(self.camera.read.return_value)
        self.assertEqual(self.detector_factory.call_args.kwargs["inference_size"], (480, 270))
        self.assertEqual(self.repository.writes, [])
        self.opencv_display.return_value.close.assert_called_once()
        self.assert_closed()

    def test_picamera2_drm_uses_existing_adapter_and_overlay_display(self) -> None:
        config = replace(self.config, camera_backend="picamera2", orientation="rotate180", requested_camera_fps=None)
        cli.run_capture(Settings(), config, FakeDatasetRepository(config), "drm", Mock())
        self.pi_factory.assert_called_once_with(480, 270, display_width=848, display_height=480, orientation="rotate180")
        self.camera_factory.assert_not_called()
        self.assertEqual(self.drm_display.call_args.args[2:4], (848, 480))
        self.drm_display.return_value.close.assert_called_once()
        self.opencv_display.assert_not_called()
        self.assert_closed()

    def test_no_display_and_ctrl_c_close_resources_without_presentation(self) -> None:
        self.controls.poll.side_effect = KeyboardInterrupt
        cli.run_capture(Settings(), self.config, self.repository, "none", Mock())
        self.opencv_display.assert_not_called()
        self.drm_display.assert_not_called()
        self.assertIn("interrupted", self.output.getvalue())
        self.assert_closed()

    def test_camera_initialization_failure_closes_detector_and_restores_terminal(self) -> None:
        self.camera_factory.side_effect = RuntimeError("camera failed")
        with self.assertRaisesRegex(RuntimeError, "camera failed"):
            cli.run_capture(Settings(), self.config, self.repository)
        self.detector.close.assert_called_once()
        self.controls_factory.return_value.__exit__.assert_called_once()

    def test_display_failure_closes_camera_detector_and_restores_terminal(self) -> None:
        self.drm_display.side_effect = RuntimeError("display failed")
        config = replace(self.config, camera_backend="picamera2", requested_camera_fps=None)
        with self.assertRaisesRegex(RuntimeError, "display failed"):
            cli.run_capture(Settings(), config, FakeDatasetRepository(config), "drm")
        self.assert_closed()

    def test_completed_session_returns_before_opening_hardware(self) -> None:
        cli.run_capture(Settings(), self.config, FakeDatasetRepository(self.config, CONDITIONS))
        self.controls_factory.assert_not_called()
        self.detector_factory.assert_not_called()
        self.camera_factory.assert_not_called()
        self.assertIn("already complete", self.output.getvalue())

    def test_incompatible_session_fails_before_opening_hardware(self) -> None:
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            cli.run_capture(Settings(), replace(self.config, inference_width=640), self.repository)
        self.controls_factory.assert_not_called()
        self.detector_factory.assert_not_called()

    def test_full_fake_session_has_explicit_stage_gates_and_no_raw_array_output(self) -> None:
        config = replace(self.config, protocol=CaptureProtocol(0, 1, 0))
        repository = FakeDatasetRepository(config)
        timestamp = 0

        def snapshot() -> DetectorSnapshot:
            nonlocal timestamp
            timestamp += 1
            return DetectorSnapshot(LandmarkResult((raw_face(),), timestamp, 10))

        self.detector.snapshot.side_effect = snapshot
        self.controls.poll.return_value = " "
        cli.run_capture(Settings(), config, repository, "none")
        self.assertEqual(len(repository.writes), 13)
        self.assertEqual([condition for condition, _ in repository.writes], list(CONDITIONS))
        self.assertEqual(self.camera.read.call_count, 26)  # Every next condition requires a separate SPACE iteration.
        output = self.output.getvalue()
        self.assertIn("CONDITION 1/13 — CENTER", output)
        self.assertIn("CONDITION 13/13 — PITCH DOWN", output)
        self.assertIn("Session complete", output)
        self.assertNotIn(str(raw_face().points[0].x), output)
        self.assert_closed()

    def test_cli_builds_source_metadata_and_configurable_protocol(self) -> None:
        with patch.object(cli, "run_capture") as run, patch.object(cli, "model_digest", return_value="a" * 64):
            cli.main(["--person-id", "p003", "--session-id", "session_02", "--camera", "picamera2",
                      "--display", "drm", "--orientation", "rotate180", "--warmup-results", "6",
                      "--samples-per-condition", "21", "--minimum-sample-interval-ms", "300"])
        config = run.call_args.args[1]
        self.assertEqual((config.person_id, config.session_id), ("p003", "session_02"))
        self.assertEqual(config.camera_backend, "picamera2")
        self.assertIsNone(config.requested_camera_fps)
        self.assertEqual(config.protocol, CaptureProtocol(6, 21, 300))
        self.assertEqual((config.inference_width, config.inference_height), (480, 270))

    def test_invalid_identifiers_and_backend_combinations_are_rejected_early(self) -> None:
        arguments = (["--person-id", "../p001", "--session-id", "s1"],
                     ["--person-id", "p001", "--session-id", ".."],
                     ["--person-id", "p001", "--session-id", "s1", "--display", "drm"],
                     ["--person-id", "p001", "--session-id", "s1", "--camera", "picamera2", "--inference-width", "481"])
        for args in arguments:
            with self.subTest(args=args), patch.object(cli, "run_capture") as run, patch("sys.stderr"):
                with self.assertRaises(SystemExit):
                    cli.main(args)
                run.assert_not_called()
