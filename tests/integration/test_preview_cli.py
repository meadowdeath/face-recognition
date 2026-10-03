import unittest
from unittest.mock import Mock, patch

import numpy as np

from face_recognition.config.settings import Settings
from face_recognition.domain.models.landmark_result import DetectorSnapshot
from face_recognition.presentation.cli import recognize as cli


class PreviewCliTests(unittest.TestCase):
    def test_q_exits_and_releases_all_resources(self) -> None:
        camera = Mock()
        detector = Mock()
        camera.read.return_value = np.zeros((4, 4, 3), dtype=np.uint8)
        detector.snapshot.return_value = DetectorSnapshot()
        renderer = Mock()
        with patch.object(cli, "OpenCVCamera", return_value=camera), \
             patch.object(cli, "MediaPipeFaceLandmarker", return_value=detector), \
             patch.object(cli.cv2, "imshow") as show, \
             patch.object(cli.cv2, "waitKey", return_value=ord("q")), \
             patch.object(cli.cv2, "destroyAllWindows") as destroy:
            cli.run_preview(Settings(), renderer)
        detector.submit.assert_called_once()
        show.assert_called_once()
        detector.close.assert_called_once()
        camera.close.assert_called_once()
        destroy.assert_called_once()

    def test_inference_submission_error_still_cleans_up(self) -> None:
        camera = Mock()
        detector = Mock()
        camera.read.return_value = object()
        detector.submit.side_effect = RuntimeError("submission failed")
        with patch.object(cli, "OpenCVCamera", return_value=camera), \
             patch.object(cli, "MediaPipeFaceLandmarker", return_value=detector), \
             patch.object(cli.cv2, "destroyAllWindows") as destroy:
            with self.assertRaisesRegex(RuntimeError, "submission failed"):
                cli.run_preview(Settings(), Mock())
        detector.close.assert_called_once()
        camera.close.assert_called_once()
        destroy.assert_called_once()

    def test_selected_mode_requires_explicit_indices_before_opening_camera(self) -> None:
        with patch.object(cli, "run_preview") as run, patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                cli.main(["--landmarks", "selected"])
        run.assert_not_called()
        with patch.object(cli, "run_preview") as run:
            cli.main(["--landmarks", "selected", "--indices", "1,4,1"])
        self.assertEqual(run.call_args.args[1].selected_indices, (1, 4))
