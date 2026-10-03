"""Shared asynchronous landmark preview for OpenCV and Picamera2 cameras."""

import argparse
from collections.abc import Sequence
from dataclasses import replace
from typing import cast

from face_recognition.application.recognize_face import LandmarkPreview
from face_recognition.application.performance import PerformanceTracker
from face_recognition.config.settings import CAMERA_ORIENTATIONS, Settings
from face_recognition.domain.interfaces.camera import Camera
from face_recognition.infrastructure.camera.opencv_camera import OpenCVCamera
from face_recognition.infrastructure.camera.picamera2_camera import Picamera2Camera
from face_recognition.infrastructure.mediapipe.face_landmarker import MediaPipeFaceLandmarker
from face_recognition.infrastructure.mediapipe.video_worker import MediaPipeVideoWorker
from face_recognition.presentation.visualization.frame_renderer import FrameRenderer
from face_recognition.presentation.visualization.display import (
    DRMDisplay,
    NoDisplay,
    OpenCVDisplay,
    OverlayTarget,
    PreviewDisplay,
)


def _parse_indices(value: str) -> tuple[int, ...]:
    try:
        indices = tuple(dict.fromkeys(int(index.strip()) for index in value.split(",")))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated landmark indices") from exc
    if any(index < 0 for index in indices):
        raise argparse.ArgumentTypeError("Landmark indices must be nonnegative")
    return indices


def _positive_dimension(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Dimensions must be positive integers")
    return number


def main(argv: Sequence[str] | None = None) -> None:
    settings = Settings()
    parser = argparse.ArgumentParser(description="Asynchronous webcam landmark preview")
    parser.add_argument("--camera", choices=("opencv", "picamera2"), default="opencv",
                        help="Camera backend (default: opencv)")
    parser.add_argument("--display", choices=("opencv", "drm", "none"), default="opencv",
                        help="Display backend (default: opencv; DRM requires Picamera2)")
    parser.add_argument("--orientation", choices=CAMERA_ORIENTATIONS, default=settings.camera_orientation)
    parser.add_argument("--inference-mode", choices=("live-stream", "video-worker"),
                        default=settings.inference_mode, help="Detector strategy (default: %(default)s)")
    parser.add_argument("--inference-width", type=_positive_dimension, default=settings.inference_width,
                        help="MediaPipe input width (default: %(default)s)")
    parser.add_argument("--inference-height", type=_positive_dimension, default=settings.inference_height,
                        help="MediaPipe input height (default: %(default)s)")
    parser.add_argument("--display-width", type=_positive_dimension, default=settings.display_width,
                        help="Capture preview/DRM width (default: %(default)s)")
    parser.add_argument("--display-height", type=_positive_dimension, default=settings.display_height,
                        help="Capture preview/DRM height (default: %(default)s)")
    parser.add_argument("--landmarks", choices=FrameRenderer.MODES, default=settings.renderer_mode)
    parser.add_argument(
        "--indices",
        type=_parse_indices,
        default=settings.renderer_landmark_indices,
        help="Explicit comma-separated landmark indices for selected rendering",
    )
    args = parser.parse_args(argv)
    if args.display == "drm" and args.camera != "picamera2":
        parser.error("--display drm requires --camera picamera2")
    if args.landmarks == "selected" and not args.indices:
        parser.error("--landmarks selected requires --indices or configured indices")
    settings = replace(
        settings, camera_orientation=args.orientation,
        inference_mode=args.inference_mode,
        inference_width=args.inference_width, inference_height=args.inference_height,
        display_width=args.display_width, display_height=args.display_height,
    )
    if args.camera == "picamera2":
        dimensions = (settings.inference_width, settings.inference_height,
                      settings.display_width, settings.display_height)
        if any(value % 2 for value in dimensions):
            parser.error("Picamera2 stream dimensions must be even")
        if settings.inference_width > settings.display_width or settings.inference_height > settings.display_height:
            parser.error("Picamera2 inference dimensions must not exceed display dimensions")
    renderer = FrameRenderer(args.landmarks, args.indices)
    run_preview(settings, renderer, camera_backend=args.camera, display_backend=args.display)


def run_preview(
    settings: Settings,
    renderer: FrameRenderer,
    camera_backend: str = "opencv",
    display_backend: str = "opencv",
) -> None:
    if display_backend not in ("opencv", "drm", "none"):
        raise ValueError(f"Unknown display backend: {display_backend}")
    if display_backend == "drm" and camera_backend != "picamera2":
        raise ValueError("DRM display requires the Picamera2 camera backend")
    performance = PerformanceTracker(settings.metrics_interval_seconds)
    if settings.inference_mode == "live-stream":
        detector_factory = MediaPipeFaceLandmarker
    elif settings.inference_mode == "video-worker":
        detector_factory = MediaPipeVideoWorker
    else:
        raise ValueError(f"Unknown inference mode: {settings.inference_mode}")
    detector = detector_factory(
        model_path=settings.landmarker_model_path,
        max_faces=settings.max_faces,
        detection_confidence=settings.detection_confidence,
        tracking_confidence=settings.tracking_confidence,
        inference_size=(settings.inference_width, settings.inference_height),
    )
    try:
        camera: Camera
        if camera_backend == "opencv":
            camera = OpenCVCamera(
                settings.camera_index, settings.display_width, settings.display_height, settings.camera_fps,
                orientation=settings.camera_orientation,
            )
        elif camera_backend == "picamera2":
            camera = Picamera2Camera(
                settings.inference_width, settings.inference_height,
                display_width=settings.display_width, display_height=settings.display_height,
                orientation=settings.camera_orientation,
            )
        else:
            raise ValueError(f"Unknown camera backend: {camera_backend}")
    except BaseException:
        detector.close()
        raise

    preview = LandmarkPreview(camera, detector, performance)
    display: PreviewDisplay | None = None
    try:
        if display_backend == "opencv":
            display = OpenCVDisplay(renderer)
        elif display_backend == "drm":
            display = DRMDisplay(
                cast(OverlayTarget, camera), renderer, settings.display_width, settings.display_height,
                interval_seconds=settings.metrics_interval_seconds,
            )
        else:
            display = NoDisplay(renderer, settings.metrics_interval_seconds)
        while True:
            result = preview.next_frame()
            if result is None:
                raise RuntimeError("Camera stopped returning frames")
            metrics = performance.snapshot(result.detector_snapshot)
            keep_running = display.show(result, metrics)
            if display_backend == "opencv":
                performance.record_display()
            elif display_backend == "drm" and cast(DRMDisplay, display).overlay_updated:
                performance.record_overlay_update()
            if not keep_running:
                break
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if display is not None:
                display.close()
        finally:
            preview.close()


if __name__ == "__main__":
    main()
