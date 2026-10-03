from pathlib import Path
from types import SimpleNamespace
from threading import Event, Thread
import unittest
from unittest.mock import Mock, patch

import numpy as np

from face_recognition.infrastructure.mediapipe import face_landmarker as module
from face_recognition.application.recognize_face import LandmarkPreview
from face_recognition.domain.models.frame import Frame, PixelFormat


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
        self.assertEqual(timestamps, [100])
        initial = self.adapter.snapshot()
        self.assertIsNone(initial.result)
        self.assertEqual((initial.submitted_frames, initial.completed_frames, initial.skipped_busy_frames), (1, 0, 1))
        self.native.detect_for_video.assert_not_called()

    def test_accepted_bgr_frame_performs_one_conversion(self) -> None:
        pixels = np.full((4, 4, 3), (10, 20, 30), dtype=np.uint8)
        frame = Frame(pixels, 4, 4, PixelFormat.BGR)
        with patch.object(module.cv2, "cvtColor", wraps=module.cv2.cvtColor) as convert:
            self.adapter.submit(frame)
        convert.assert_called_once()
        self.assertIs(convert.call_args.args[0], pixels)
        self.assertEqual(convert.call_args.args[1], module.cv2.COLOR_BGR2RGB)
        image = self.native.detect_async.call_args.args[0]
        np.testing.assert_array_equal(image.numpy_view(), pixels[:, :, ::-1])

    def test_accepted_yuv_frame_converts_directly_to_rgb_and_crops_padding(self) -> None:
        width, height, stride = 480, 270, 512
        # Distinct colors at both edges catch bad U/V plane or padding cropping.
        source = np.full((height, width, 3), (200, 20, 10), dtype=np.uint8)
        source[:, :width // 2] = (10, 200, 30)
        packed = module.cv2.cvtColor(source, module.cv2.COLOR_RGB2YUV_I420)
        expected = module.cv2.cvtColor(packed, module.cv2.COLOR_YUV2RGB_I420)
        flat = packed.reshape(-1)
        luma_end = width * height
        chroma_size = luma_end // 4
        yuv = np.full((height * 3 // 2, stride), 128, dtype=np.uint8)
        padded = yuv.reshape(-1)
        padded[:height * stride].reshape(height, stride)[:, :width] = flat[:luma_end].reshape(height, width)
        u_start = height * stride
        v_start = u_start + height * stride // 4
        for start, plane in ((u_start, flat[luma_end:luma_end + chroma_size]),
                             (v_start, flat[luma_end + chroma_size:])):
            padded[start:start + height * stride // 4].reshape(height // 2, stride // 2)[:, :width // 2] = plane.reshape(height // 2, width // 2)
        frame = Frame(yuv, width, height, PixelFormat.YUV420_I420)
        self.adapter._inference_size = (width, height)
        with patch.object(module.cv2, "cvtColor", wraps=module.cv2.cvtColor) as convert, \
             patch.object(module.cv2, "resize") as resize:
            self.adapter.submit(frame)
        convert.assert_called_once()
        self.assertIs(convert.call_args.args[0], yuv)
        self.assertEqual(convert.call_args.args[1], module.cv2.COLOR_YUV2RGB_I420)
        resize.assert_not_called()
        image = self.native.detect_async.call_args.args[0]
        self.assertEqual(image.numpy_view().shape, (270, 480, 3))
        np.testing.assert_array_equal(image.numpy_view(), expected)
        self.assertEqual(self.adapter.snapshot().submitted_frames, 1)

    def test_busy_bgr_and_yuv_frames_do_not_convert_resize_or_construct_image(self) -> None:
        self.adapter.submit(self.frame)
        # Opaque data would fail preprocessing if the gate did not return first.
        frames = (Frame(object(), 848, 480, PixelFormat.BGR),
                  Frame(object(), 480, 270, PixelFormat.YUV420_I420))
        with patch.object(module.cv2, "cvtColor") as convert, \
             patch.object(module.cv2, "resize") as resize, \
             patch.object(module.mp, "Image") as image:
            for frame in frames:
                self.adapter.submit(frame)
        convert.assert_not_called()
        resize.assert_not_called()
        image.assert_not_called()
        self.native.detect_async.assert_called_once()
        status = self.adapter.snapshot()
        self.assertEqual((status.submitted_frames, status.completed_frames, status.skipped_busy_frames), (1, 0, 2))
        timestamp = self.native.detect_async.call_args.args[1]
        self.adapter._on_result(SimpleNamespace(face_landmarks=[]), None, timestamp)
        self.adapter.submit(Frame(self.frame, 4, 4, PixelFormat.BGR))
        status = self.adapter.snapshot()
        self.assertEqual((status.submitted_frames, status.completed_frames, status.skipped_busy_frames), (2, 1, 2))

    def test_latest_result_counts_latency_and_out_of_order_callback(self) -> None:
        with patch.object(module, "monotonic_ns", return_value=100_000_000):
            for _ in range(3):
                self.adapter.submit(self.frame)
        self.publish(100)
        with patch.object(module, "monotonic_ns", return_value=100_000_000):
            self.adapter.submit(self.frame)
        self.publish(101)
        first = self.adapter.snapshot()
        self.assertEqual(first.result.timestamp_ms, 101)
        self.assertEqual(first.result.latency_ms, 39.0)
        self.assertEqual(len(first.result.faces), 1)
        self.assertIs(first.result, self.adapter.snapshot().result)
        with patch.object(module, "monotonic_ns", return_value=100_000_000):
            self.adapter.submit(self.frame)
        self.publish(100)  # Stale callbacks cannot release the current request.
        status = self.adapter.snapshot()
        self.assertIs(status.result, first.result)
        self.assertEqual((status.submitted_frames, status.completed_frames), (3, 2))
        self.assertEqual(status.skipped_busy_frames, 2)
        self.assertTrue(self.adapter._in_flight)
        self.publish(102, faces=False)
        self.assertEqual(self.adapter.snapshot().result.faces, ())
        self.assertFalse(self.adapter._in_flight)

    def test_callback_does_not_convert_landmarks(self) -> None:
        class RawResult:
            accesses = 0

            @property
            def face_landmarks(self) -> list:
                self.accesses += 1
                return []

        raw = RawResult()
        with patch.object(module, "monotonic_ns", return_value=100_000_000):
            self.adapter.submit(self.frame)
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
        self.assertFalse(self.adapter._in_flight)
        self.native.detect_async.side_effect = None
        self.adapter.submit(self.frame)
        self.assertEqual(self.adapter.snapshot().submitted_frames, 1)

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
            timestamp = self.native.detect_async.call_args.args[1]
            adapter._on_result(SimpleNamespace(face_landmarks=[]), None, timestamp)
            with patch.object(module.cv2, "resize") as resize:
                adapter.submit(np.zeros((270, 480, 3), dtype=np.uint8))
                resize.assert_not_called()
        finally:
            adapter.close()

    def test_close_handles_late_callback_and_is_idempotent(self) -> None:
        with patch.object(module, "monotonic_ns", return_value=100_000_000):
            self.adapter.submit(self.frame)
        self.publish(100)
        with patch.object(module, "monotonic_ns", return_value=100_000_000):
            self.adapter.submit(self.frame)
        self.native.close.side_effect = lambda: self.publish(101)
        self.adapter.close()
        self.adapter.close()
        self.native.close.assert_called_once()
        self.assertIsNone(self.adapter._latest_raw)
        self.assertIsNone(self.adapter._cached_result)
        self.assertFalse(self.adapter._in_flight)
        self.assertEqual(self.adapter._completed_frames, 1)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.adapter.submit(self.frame)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.adapter.snapshot()

    def test_busy_frames_skip_all_preprocessing_and_next_submission_is_current(self) -> None:
        self.adapter._inference_size = (2, 2)
        with patch.object(module, "monotonic_ns", return_value=100_000_000):
            self.adapter.submit(self.frame)
        with patch.object(module.cv2, "resize") as resize, \
             patch.object(module.cv2, "cvtColor") as convert, \
             patch.object(module.mp, "Image") as image:
            for _ in range(3):
                self.adapter.submit(object())  # Even shape must not be accessed.
        resize.assert_not_called()
        convert.assert_not_called()
        image.assert_not_called()
        self.native.detect_async.assert_called_once()
        status = self.adapter.snapshot()
        self.assertEqual((status.submitted_frames, status.completed_frames, status.skipped_busy_frames), (1, 0, 3))
        self.publish(100, faces=False)
        newest = np.full((2, 2, 3), (1, 2, 3), dtype=np.uint8)
        self.adapter.submit(newest)
        image = self.native.detect_async.call_args.args[0]
        np.testing.assert_array_equal(image.numpy_view(), newest[:, :, ::-1])
        status = self.adapter.snapshot()
        self.assertEqual((status.submitted_frames, status.completed_frames, status.skipped_busy_frames), (2, 1, 3))

    def test_each_preprocessing_exception_releases_slot(self) -> None:
        self.adapter._inference_size = (2, 2)
        for target in ((module.cv2, "resize"), (module.cv2, "cvtColor"), (module.mp, "Image")):
            with self.subTest(operation=target[1]):
                before = self.adapter.snapshot().submitted_frames
                with patch.object(*target, side_effect=RuntimeError("preprocessing failed")):
                    with self.assertRaisesRegex(RuntimeError, "preprocessing failed"):
                        self.adapter.submit(self.frame)
                self.assertFalse(self.adapter._in_flight)
                self.assertEqual(self.adapter.snapshot().submitted_frames, before)
                self.adapter.submit(self.frame)
                timestamp = self.native.detect_async.call_args.args[1]
                self.adapter._on_result(SimpleNamespace(face_landmarks=[]), None, timestamp)
                self.assertEqual(self.adapter.snapshot().submitted_frames, before + 1)

    def test_application_keeps_current_capture_frames_while_detector_is_busy(self) -> None:
        frames = [np.full((4, 4, 3), value, dtype=np.uint8) for value in range(4)]
        camera = Mock()
        camera.read.side_effect = frames
        preview = LandmarkPreview(camera, self.adapter)
        for frame in frames[:3]:
            self.assertIs(preview.next_frame().frame, frame)
        self.native.detect_async.assert_called_once()
        timestamp = self.native.detect_async.call_args.args[1]
        self.adapter._on_result(SimpleNamespace(face_landmarks=[]), None, timestamp)
        self.assertIs(preview.next_frame().frame, frames[3])
        image = self.native.detect_async.call_args.args[0]
        np.testing.assert_array_equal(image.numpy_view(), frames[3])
        metrics = preview.performance.snapshot(self.adapter.snapshot())
        self.assertEqual((metrics.captured_frames, metrics.submitted_inference_frames,
                          metrics.completed_inference_frames, metrics.skipped_busy_frames), (4, 2, 1, 2))
        preview.close()
        camera.close.assert_called_once()

    def test_inline_callback_does_not_deadlock_or_allow_overlapping_dispatch(self) -> None:
        def dispatch(image, timestamp):
            self.options.result_callback(SimpleNamespace(face_landmarks=[]), image, timestamp)
            self.adapter.submit(object())  # Still inside the native API call.

        self.native.detect_async.side_effect = dispatch
        self.adapter.submit(self.frame)
        self.adapter.submit(self.frame)
        status = self.adapter.snapshot()
        self.assertEqual((status.submitted_frames, status.completed_frames, status.skipped_busy_frames), (2, 2, 2))

    def test_callback_latency_is_fixed_but_result_age_grows(self) -> None:
        with patch.object(module, "monotonic_ns", return_value=100_000_000):
            self.adapter.submit(self.frame)
        self.publish(100)
        with patch.object(module, "monotonic_ns", return_value=200_000_000):
            first = self.adapter.snapshot()
        with patch.object(module, "monotonic_ns", return_value=500_000_000):
            later = self.adapter.snapshot()
        self.assertEqual(first.result.latency_ms, 40.0)
        self.assertEqual(later.result.latency_ms, 40.0)
        self.assertEqual((first.result_age_ms, later.result_age_ms), (100.0, 400.0))

    def test_concurrent_close_waits_for_dispatch_and_ignores_its_callback(self) -> None:
        entered, release, closing = Event(), Event(), Event()
        errors = []

        def dispatch(image, timestamp):
            entered.set()
            if not release.wait(2):
                raise AssertionError("Test did not release dispatch")
            self.options.result_callback(SimpleNamespace(face_landmarks=[]), image, timestamp)

        def run(operation):
            try:
                operation()
            except BaseException as exc:
                errors.append(exc)

        wait = self.adapter._dispatch_finished.wait

        def waiting():
            closing.set()
            return wait()

        self.native.detect_async.side_effect = dispatch
        submitter = Thread(target=lambda: run(lambda: self.adapter.submit(self.frame)))
        closer = Thread(target=lambda: run(self.adapter.close))
        with patch.object(self.adapter._dispatch_finished, "wait", side_effect=waiting):
            try:
                submitter.start()
                self.assertTrue(entered.wait(2))
                closer.start()
                self.assertTrue(closing.wait(2))
                self.native.close.assert_not_called()
            finally:
                release.set()
                submitter.join(2)
                if closer.ident is not None:
                    closer.join(2)
        self.assertFalse(submitter.is_alive() or closer.is_alive())
        self.assertEqual(errors, [])
        self.native.close.assert_called_once()
        self.assertEqual(self.adapter._completed_frames, 0)
        self.assertIsNone(self.adapter._latest_raw)
        self.assertFalse(self.adapter._in_flight)
