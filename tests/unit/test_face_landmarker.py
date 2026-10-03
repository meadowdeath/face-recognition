from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from face_recognition.infrastructure.mediapipe import face_landmarker as module


class FaceLandmarkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = Mock(spec=Path)
        self.model.is_file.return_value = True
        self.native = Mock()
        self.factory = patch.object(
            module.mp.tasks.vision.FaceLandmarker, "create_from_options", return_value=self.native
        )
        factory = self.factory.start()
        self.adapter = module.MediaPipeFaceLandmarker(self.model, 1, 0.5, 0.5)
        self.options = factory.call_args.args[0]
        self.frame = np.zeros((4, 4, 3), dtype=np.uint8)

    def tearDown(self) -> None:
        self.adapter.close()
        self.factory.stop()

    def publish(self, timestamp_ms: int, faces: bool = True) -> None:
        points = [[SimpleNamespace(x=0.1, y=0.2, z=0.3)]] if faces else []
        with patch.object(module, "monotonic_ns", return_value=140_000_000):
            self.options.result_callback(SimpleNamespace(face_landmarks=points), None, timestamp_ms)

    def test_live_stream_options_and_nonblocking_submission(self) -> None:
        self.assertEqual(self.options.running_mode, module.mp.tasks.vision.RunningMode.LIVE_STREAM)
        self.assertEqual(self.options.num_faces, 1)
        self.assertFalse(self.options.output_face_blendshapes)
        self.assertFalse(self.options.output_facial_transformation_matrixes)
        with patch.object(module, "monotonic_ns", return_value=100_000_000):
            self.adapter.submit(self.frame)
            self.adapter.submit(self.frame)
        timestamps = [call.args[1] for call in self.native.detect_async.call_args_list]
        self.assertEqual(timestamps, [100, 101])
        initial = self.adapter.snapshot()
        self.assertIsNone(initial.result)
        self.assertEqual((initial.submitted_frames, initial.completed_frames), (2, 0))
        self.native.detect_for_video.assert_not_called()

    def test_latest_result_counts_latency_and_out_of_order_callback(self) -> None:
        with patch.object(module, "monotonic_ns", return_value=100_000_000):
            for _ in range(3):
                self.adapter.submit(self.frame)
        self.publish(101)
        first = self.adapter.snapshot()
        self.assertEqual(first.result.timestamp_ms, 101)
        self.assertEqual(first.result.latency_ms, 39.0)
        self.assertEqual(len(first.result.faces), 1)
        self.assertIs(first.result, self.adapter.snapshot().result)
        self.publish(100)  # Cannot replace a more recent result.
        status = self.adapter.snapshot()
        self.assertIs(status.result, first.result)
        self.assertEqual((status.submitted_frames, status.completed_frames), (3, 2))
        self.publish(102, faces=False)
        self.assertEqual(self.adapter.snapshot().result.faces, ())

    def test_callback_does_not_convert_landmarks(self) -> None:
        class RawResult:
            accesses = 0

            @property
            def face_landmarks(self) -> list:
                self.accesses += 1
                return []

        raw = RawResult()
        self.options.result_callback(raw, None, 100)
        self.assertEqual(raw.accesses, 0)
        self.adapter.snapshot()
        self.adapter.snapshot()
        self.assertEqual(raw.accesses, 1)

    def test_failed_submission_does_not_increase_count(self) -> None:
        self.native.detect_async.side_effect = RuntimeError("native failure")
        with self.assertRaisesRegex(RuntimeError, "native failure"):
            self.adapter.submit(self.frame)
        self.assertEqual(self.adapter.snapshot().submitted_frames, 0)

    def test_preview_input_is_resized_only_when_inference_dimensions_differ(self) -> None:
        adapter = module.MediaPipeFaceLandmarker(self.model, 1, 0.5, 0.5, inference_size=(480, 270))
        try:
            frame = np.zeros((480, 848, 3), dtype=np.uint8)
            with patch.object(module.cv2, "resize", wraps=module.cv2.resize) as resize:
                adapter.submit(frame)
                image = self.native.detect_async.call_args.args[0]
                self.assertEqual(image.numpy_view().shape, (270, 480, 3))
                resize.assert_called_once()
                self.assertEqual(frame.shape, (480, 848, 3))
            with patch.object(module.cv2, "resize") as resize:
                adapter.submit(np.zeros((270, 480, 3), dtype=np.uint8))
                resize.assert_not_called()
        finally:
            adapter.close()

    def test_close_handles_late_callback_and_is_idempotent(self) -> None:
        self.publish(100)
        self.native.close.side_effect = lambda: self.publish(101)
        self.adapter.close()
        self.adapter.close()
        self.native.close.assert_called_once()
        self.assertIsNone(self.adapter._latest_raw)
        self.assertIsNone(self.adapter._cached_result)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.adapter.submit(self.frame)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.adapter.snapshot()
