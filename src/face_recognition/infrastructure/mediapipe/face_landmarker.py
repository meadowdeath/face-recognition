"""MediaPipe live-stream inference with a single latest-result slot.

submit(), snapshot(), and close() are called by the main loop. Only the
MediaPipe callback runs on another thread; no frame queue is maintained here.
"""

from pathlib import Path
from threading import Condition, Lock
from time import monotonic_ns
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from face_recognition.domain.models.face_landmarks import FaceLandmarks, Landmark
from face_recognition.domain.models.landmark_result import DetectorSnapshot, LandmarkResult


class MediaPipeFaceLandmarker:
    def __init__(
        self,
        model_path: Path,
        max_faces: int,
        detection_confidence: float,
        tracking_confidence: float,
        *,
        inference_size: tuple[int, int] | None = None,
    ) -> None:
        if inference_size is not None and any(value <= 0 for value in inference_size):
            raise ValueError("Inference dimensions must be positive")
        self._inference_size = inference_size
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Face Landmarker model missing: {model_path}. "
                "Run python tools/download_face_landmarker.py first."
            )
        self._lock = Lock()
        self._dispatch_finished = Condition(self._lock)
        self._closed = False
        self._in_flight = False
        self._dispatching = False
        self._active_timestamp_ms: int | None = None
        self._latest_raw: tuple[Any, int, float] | None = None
        self._cached_result: LandmarkResult | None = None
        self._submitted_frames = 0
        self._completed_frames = 0
        self._skipped_busy_frames = 0
        self._last_timestamp_ms = -1
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            num_faces=max_faces,
            min_face_detection_confidence=detection_confidence,
            min_face_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            result_callback=self._on_result,
        )
        self._detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def submit(self, frame: np.ndarray) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Face landmarker is closed")
            # Reserve before touching pixels. Dispatch remains busy even if a
            # callback arrives before detect_async() has returned.
            if self._in_flight or self._dispatching:
                self._skipped_busy_frames += 1
                return
            timestamp_ms = max(monotonic_ns() // 1_000_000, self._last_timestamp_ms + 1)
            self._last_timestamp_ms = timestamp_ms
            self._active_timestamp_ms = timestamp_ms
            self._in_flight = True
            self._dispatching = True
        try:
            if self._inference_size is not None and (frame.shape[1], frame.shape[0]) != self._inference_size:
                frame = cv2.resize(frame, self._inference_size, interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            # Never hold the callback/state lock across a MediaPipe call.
            self._detector.detect_async(image, timestamp_ms)
            with self._lock:
                self._submitted_frames += 1
        except BaseException:
            with self._lock:
                self._in_flight = False
                self._active_timestamp_ms = None
            raise
        finally:
            with self._dispatch_finished:
                self._dispatching = False
                self._dispatch_finished.notify_all()

    def _on_result(self, result: Any, image: Any, timestamp_ms: int) -> None:
        completed_ms = monotonic_ns() / 1_000_000
        with self._lock:
            if self._closed or timestamp_ms != self._active_timestamp_ms:
                return
            self._completed_frames += 1
            if self._latest_raw is None or timestamp_ms > self._latest_raw[1]:
                self._latest_raw = (result, timestamp_ms, completed_ms)
            self._in_flight = False
            self._active_timestamp_ms = None

    def snapshot(self) -> DetectorSnapshot:
        with self._lock:
            if self._closed:
                raise RuntimeError("Face landmarker is closed")
            latest = self._latest_raw
            submitted = self._submitted_frames
            completed = self._completed_frames
            skipped = self._skipped_busy_frames
        if latest is not None:
            raw, timestamp_ms, completed_ms = latest
            if self._cached_result is None or self._cached_result.timestamp_ms != timestamp_ms:
                # Convert once per new result, outside both callback and lock.
                self._cached_result = LandmarkResult(
                    faces=tuple(
                        FaceLandmarks(tuple(Landmark(p.x, p.y, p.z) for p in face))
                        for face in raw.face_landmarks
                    ),
                    timestamp_ms=timestamp_ms,
                    latency_ms=max(0.0, completed_ms - timestamp_ms),
                )
        result_age_ms = None if latest is None else max(0.0, monotonic_ns() / 1_000_000 - latest[1])
        return DetectorSnapshot(self._cached_result, submitted, completed, skipped, result_age_ms)

    def close(self) -> None:
        with self._dispatch_finished:
            if self._closed:
                return
            self._closed = True
            self._in_flight = False
            self._active_timestamp_ms = None
            # A concurrent close must not dispose MediaPipe during preprocessing
            # or dispatch. Condition.wait releases the lock used by callbacks.
            while self._dispatching:
                self._dispatch_finished.wait()
        try:
            # Do not hold the lock while MediaPipe drains/stops its callbacks.
            self._detector.close()
        finally:
            with self._lock:
                self._latest_raw = None
            self._cached_result = None
