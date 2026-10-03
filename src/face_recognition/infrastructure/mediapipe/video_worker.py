"""Experimental VIDEO inference: one worker and one replaceable pending frame."""

from pathlib import Path
from threading import Condition, Lock, Thread
from time import monotonic_ns
from typing import Any

import mediapipe as mp
import numpy as np

from face_recognition.domain.models.face_landmarks import FaceLandmarks, Landmark
from face_recognition.domain.models.frame import Frame
from face_recognition.domain.models.landmark_result import DetectorSnapshot, LandmarkResult
from face_recognition.infrastructure.mediapipe.frame_preprocessing import prepare_rgb


class MediaPipeVideoWorker:
    def __init__(
        self, model_path: Path, max_faces: int, detection_confidence: float,
        tracking_confidence: float, *, inference_size: tuple[int, int] | None = None,
    ) -> None:
        if inference_size is not None and any(value <= 0 for value in inference_size):
            raise ValueError("Inference dimensions must be positive")
        if not model_path.is_file():
            raise FileNotFoundError(f"Face Landmarker model missing: {model_path}. Run python tools/download_face_landmarker.py first.")
        self._inference_size = inference_size
        self._condition = Condition()
        self._close_lock = Lock()
        self._closed = False
        self._stopping = False
        self._error: BaseException | None = None
        self._pending: tuple[Frame | np.ndarray, int] | None = None
        self._latest_raw: tuple[Any, int, float, float, float] | None = None
        self._cached_result: LandmarkResult | None = None
        self._accepted = self._overwritten = self._processed = self._completed = 0
        self._last_timestamp_ms = -1
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO, num_faces=max_faces,
            min_face_detection_confidence=detection_confidence,
            min_face_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            output_face_blendshapes=False, output_facial_transformation_matrixes=False,
        )
        self._detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self._worker = Thread(target=self._run, name="mediapipe-video-worker", daemon=False)
        try:
            self._worker.start()
        except BaseException:
            self._detector.close()
            raise

    def _check_available(self) -> None:
        if self._closed:
            raise RuntimeError("Face landmarker is closed")
        if self._error is not None:
            raise RuntimeError("MediaPipe VIDEO worker failed") from self._error

    def submit(self, frame: Frame | np.ndarray) -> None:
        with self._condition:
            self._check_available()
            if self._pending is not None:
                self._overwritten += 1
            self._pending = (frame, monotonic_ns())
            self._accepted += 1
            self._condition.notify()

    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._stopping or self._pending is not None)
                    if self._stopping:
                        return
                    frame, submitted_ns = self._pending
                    self._pending = None
                started_ns = monotonic_ns()
                timestamp_ms = max(submitted_ns // 1_000_000, self._last_timestamp_ms + 1)
                self._last_timestamp_ms = timestamp_ms
                rgb = prepare_rgb(frame, self._inference_size)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
                with self._condition:
                    if self._stopping:
                        return
                    self._processed += 1
                # Only this worker calls MediaPipe; no state lock is held here.
                ready_ns = monotonic_ns()
                raw = self._detector.detect_for_video(image, timestamp_ms)
                finished_ns = monotonic_ns()
                preprocessing_ms = (ready_ns - started_ns) / 1_000_000
                sync_ms = (finished_ns - ready_ns) / 1_000_000
                total_ms = (finished_ns - started_ns) / 1_000_000
                with self._condition:
                    if self._stopping:
                        return
                    self._completed += 1
                    self._latest_raw = (raw, timestamp_ms, preprocessing_ms, sync_ms, total_ms)
                    self._condition.notify_all()
        except BaseException as exc:
            with self._condition:
                self._error = exc
                self._pending = None
                self._stopping = True
                self._condition.notify_all()

    def snapshot(self) -> DetectorSnapshot:
        with self._condition:
            self._check_available()
            latest = self._latest_raw
            accepted, overwritten = self._accepted, self._overwritten
            processed, completed = self._processed, self._completed
        preprocessing_ms = sync_ms = total_ms = age_ms = None
        if latest is not None:
            raw, timestamp_ms, preprocessing_ms, sync_ms, total_ms = latest
            if self._cached_result is None or self._cached_result.timestamp_ms != timestamp_ms:
                self._cached_result = LandmarkResult(
                    tuple(FaceLandmarks(tuple(Landmark(p.x, p.y, p.z) for p in face))
                          for face in raw.face_landmarks), timestamp_ms, total_ms,
                )
            age_ms = max(0.0, monotonic_ns() / 1_000_000 - timestamp_ms)
        return DetectorSnapshot(
            result=self._cached_result, submitted_frames=processed, completed_frames=completed,
            result_age_ms=age_ms, preprocessing_ms=preprocessing_ms,
            inference_mode="video-worker", accepted_frames=accepted,
            overwritten_pending_frames=overwritten, sync_inference_ms=sync_ms,
            total_worker_latency_ms=total_ms,
        )

    def close(self) -> None:
        with self._close_lock:
            with self._condition:
                if self._closed:
                    return
                self._closed = True
                self._stopping = True
                self._pending = None
                self._condition.notify_all()
            try:
                self._worker.join()
            finally:
                try:
                    self._detector.close()
                finally:
                    with self._condition:
                        self._latest_raw = None
                    self._cached_result = None
