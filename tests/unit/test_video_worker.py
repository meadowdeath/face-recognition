from pathlib import Path
from threading import Condition, Event, Thread, get_ident
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from face_recognition.domain.models.frame import Frame, PixelFormat
from face_recognition.infrastructure.mediapipe import video_worker as module
from face_recognition.infrastructure.mediapipe import frame_preprocessing as preprocessing


class VideoWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = Mock(spec=Path)
        self.model.is_file.return_value = True
        self.native = Mock()
        self.native.detect_for_video.return_value = SimpleNamespace(face_landmarks=[])
        self.factory = patch.object(module.mp.tasks.vision.FaceLandmarker, "create_from_options", return_value=self.native)
        self.create = self.factory.start()
        self.addCleanup(self.factory.stop)

    def make_worker(self, **kwargs):
        worker = module.MediaPipeVideoWorker(self.model, 1, 0.5, 0.5, **kwargs)
        self.addCleanup(worker.close)
        return worker

    def wait_completed(self, worker, count):
        with worker._condition:
            self.assertTrue(worker._condition.wait_for(
                lambda: worker._completed >= count or worker._error is not None, timeout=2))
            self.assertIsNone(worker._error)

    def test_one_worker_capacity_one_overwrites_and_preprocesses_only_consumed_frames(self) -> None:
        entered, release = Event(), Event()
        frames = [np.full((4, 4, 3), value, dtype=np.uint8) for value in range(4)]
        seen, identities = [], []

        def detect(image, timestamp):
            seen.append(int(image.numpy_view()[0, 0, 0]))
            identities.append(get_ident())
            if len(seen) == 1:
                entered.set()
                if not release.wait(2):
                    raise AssertionError("Inference was not released")
            return SimpleNamespace(face_landmarks=[[SimpleNamespace(x=0.5, y=0.5, z=0.0)]])

        self.native.detect_for_video.side_effect = detect
        with patch.object(module, "prepare_rgb", wraps=module.prepare_rgb) as prepare:
            worker = self.make_worker()
            try:
                worker.submit(frames[0])
                self.assertTrue(entered.wait(2))
                for frame in frames[1:]:
                    worker.submit(frame)
                with worker._condition:
                    self.assertIs(worker._pending[0], frames[-1])
                status = worker.snapshot()
                self.assertEqual((status.accepted_frames, status.submitted_frames,
                                  status.overwritten_pending_frames, status.completed_frames), (4, 1, 2, 0))
                self.assertEqual(prepare.call_count, 1)
            finally:
                release.set()
            self.wait_completed(worker, 2)
            self.assertEqual(prepare.call_count, 2)
            self.assertIs(prepare.call_args_list[1].args[0], frames[-1])
        self.assertEqual(seen, [0, 3])
        self.assertEqual(set(identities), {worker._worker.ident})
        self.assertNotEqual(worker._worker.ident, get_ident())
        self.assertIsNone(worker._pending)
        options = self.create.call_args.args[0]
        self.assertEqual(options.running_mode, module.mp.tasks.vision.RunningMode.VIDEO)
        self.assertIsNone(options.result_callback)
        self.assertFalse(options.output_face_blendshapes)
        self.assertFalse(options.output_facial_transformation_matrixes)
        self.native.detect_async.assert_not_called()
        status = worker.snapshot()
        self.assertEqual(status.inference_mode, "video-worker")
        self.assertEqual((status.accepted_frames, status.submitted_frames,
                          status.overwritten_pending_frames, status.completed_frames), (4, 2, 2, 2))
        self.assertEqual(status.result.timestamp_ms, self.native.detect_for_video.call_args.args[1])
        self.assertIs(status.result, worker.snapshot().result)
        worker.close()
        self.assertFalse(worker._worker.is_alive())
        self.assertIsNone(worker._latest_raw)
        worker.close()
        self.native.close.assert_called_once()

    def test_worker_blocks_when_idle_and_handles_spurious_wakeups(self) -> None:
        condition = Condition()
        first_wait, second_wait = Event(), Event()
        actual_wait = condition.wait
        calls = []

        def wait(timeout=None):
            calls.append(get_ident())
            (first_wait if len(calls) == 1 else second_wait).set()
            return actual_wait(timeout)

        with patch.object(module, "Condition", return_value=condition), \
             patch.object(condition, "wait", side_effect=wait):
            worker = self.make_worker()
            self.assertTrue(first_wait.wait(2))
            with condition:
                self.assertEqual(len(calls), 1)
                condition.notify_all()
            self.assertTrue(second_wait.wait(2))
            with condition:
                self.assertEqual(len(calls), 2)
                self.assertEqual(set(calls), {worker._worker.ident})
            self.native.detect_for_video.assert_not_called()
            worker.close()
        self.assertFalse(worker._worker.is_alive())
        self.native.close.assert_called_once()

    def test_timestamps_increase_even_with_identical_clock_values(self) -> None:
        with patch.object(module, "monotonic_ns", return_value=100_000_000):
            worker = self.make_worker()
            for count in range(1, 4):
                worker.submit(np.zeros((4, 4, 3), dtype=np.uint8))
                self.wait_completed(worker, count)
        self.assertEqual([call.args[1] for call in self.native.detect_for_video.call_args_list], [100, 101, 102])

    def test_exact_worker_timings_and_age_include_pending_wait_only_in_age(self) -> None:
        main_ident = get_ident()
        worker_times = iter([105_000_000, 108_500_000, 120_500_000])
        main_time = [100_000_000]

        def clock():
            return main_time[0] if get_ident() == main_ident else next(worker_times)

        with patch.object(module, "monotonic_ns", side_effect=clock):
            worker = self.make_worker()
            worker.submit(np.zeros((4, 4, 3), dtype=np.uint8))
            self.wait_completed(worker, 1)
            main_time[0] = 130_000_000
            status = worker.snapshot()
        self.assertEqual(status.preprocessing_ms, 3.5)
        self.assertEqual(status.sync_inference_ms, 12.0)
        self.assertEqual(status.total_worker_latency_ms, 15.5)
        self.assertEqual(status.result.latency_ms, 15.5)
        self.assertEqual(status.result_age_ms, 30.0)
        self.assertIsNone(status.total_callback_latency_ms)
        self.assertIsNone(status.dispatch_call_ms)
        self.assertEqual(status.skipped_busy_frames, 0)

    def test_native_yuv_is_converted_directly_only_by_worker(self) -> None:
        yuv = np.full((405, 512), 128, dtype=np.uint8)
        yuv[:270] = 16
        frame = Frame(yuv, 480, 270, PixelFormat.YUV420_I420)
        with patch.object(preprocessing.cv2, "cvtColor", wraps=preprocessing.cv2.cvtColor) as convert, \
             patch.object(preprocessing.cv2, "resize") as resize:
            worker = self.make_worker(inference_size=(480, 270))
            worker.submit(frame)
            self.wait_completed(worker, 1)
        convert.assert_called_once()
        self.assertEqual(convert.call_args.args[1], preprocessing.cv2.COLOR_YUV2RGB_I420)
        resize.assert_not_called()
        self.assertEqual(self.native.detect_for_video.call_args.args[0].numpy_view().shape, (270, 480, 3))

    def test_worker_failure_surfaces_and_cleanup_joins(self) -> None:
        self.native.detect_for_video.side_effect = ValueError("Native failure")
        worker = self.make_worker()
        worker.submit(np.zeros((4, 4, 3), dtype=np.uint8))
        with worker._condition:
            self.assertTrue(worker._condition.wait_for(lambda: worker._error is not None, timeout=2))
        for operation in (worker.snapshot, lambda: worker.submit(object())):
            with self.assertRaisesRegex(RuntimeError, "VIDEO worker failed") as error:
                operation()
            self.assertIsInstance(error.exception.__cause__, ValueError)
        worker.close()
        self.assertFalse(worker._worker.is_alive())
        self.native.close.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            worker.snapshot()

    def test_worker_start_failure_closes_native_detector(self) -> None:
        with patch.object(module.Thread, "start", side_effect=RuntimeError("Thread start failed")):
            with self.assertRaisesRegex(RuntimeError, "Thread start failed"):
                self.make_worker()
        self.native.close.assert_called_once()

    def test_close_waits_for_active_inference_and_discards_pending_frame(self) -> None:
        entered, release = Event(), Event()
        errors = []

        def detect(image, timestamp):
            entered.set()
            if not release.wait(2):
                raise AssertionError("Active inference was not released")
            return SimpleNamespace(face_landmarks=[])

        def close():
            try:
                worker.close()
            except BaseException as exc:
                errors.append(exc)

        self.native.detect_for_video.side_effect = detect
        worker = self.make_worker()
        worker.submit(np.zeros((4, 4, 3), dtype=np.uint8))
        self.assertTrue(entered.wait(2))
        worker.submit(object())  # Must be discarded, never preprocessed.
        closer = Thread(target=close)
        try:
            closer.start()
            with worker._condition:
                self.assertTrue(worker._condition.wait_for(lambda: worker._closed, timeout=2))
                self.assertIsNone(worker._pending)
            self.assertTrue(closer.is_alive())
            self.native.close.assert_not_called()
        finally:
            release.set()
            closer.join(2)
        self.assertFalse(closer.is_alive() or worker._worker.is_alive())
        self.assertEqual(errors, [])
        self.native.detect_for_video.assert_called_once()
        self.native.close.assert_called_once()
