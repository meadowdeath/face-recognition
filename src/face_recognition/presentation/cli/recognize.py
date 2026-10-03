"""Laptop webcam landmark preview; identity recognition is not implemented."""

import argparse
from collections.abc import Sequence

import cv2

from face_recognition.application.recognize_face import LandmarkPreview
from face_recognition.application.performance import PerformanceTracker
from face_recognition.config.settings import Settings
from face_recognition.infrastructure.camera.opencv_camera import OpenCVCamera
from face_recognition.infrastructure.mediapipe.face_landmarker import MediaPipeFaceLandmarker
from face_recognition.presentation.visualization.frame_renderer import FrameRenderer


def _parse_indices(value: str) -> tuple[int, ...]:
    try:
        indices = tuple(dict.fromkeys(int(index.strip()) for index in value.split(",")))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated landmark indices") from exc
    if any(index < 0 for index in indices):
        raise argparse.ArgumentTypeError("Landmark indices must be nonnegative")
    return indices


def main(argv: Sequence[str] | None = None) -> None:
    settings = Settings()
    parser = argparse.ArgumentParser(description="Asynchronous webcam landmark preview")
    parser.add_argument("--landmarks", choices=FrameRenderer.MODES, default=settings.renderer_mode)
    parser.add_argument(
        "--indices",
        type=_parse_indices,
        default=settings.renderer_landmark_indices,
        help="Explicit comma-separated landmark indices for selected rendering",
    )
    args = parser.parse_args(argv)
    if args.landmarks == "selected" and not args.indices:
        parser.error("--landmarks selected requires --indices or configured indices")
    renderer = FrameRenderer(args.landmarks, args.indices)
    run_preview(settings, renderer)


def run_preview(settings: Settings, renderer: FrameRenderer) -> None:
    performance = PerformanceTracker(settings.metrics_interval_seconds)
    detector = MediaPipeFaceLandmarker(
        model_path=settings.landmarker_model_path,
        max_faces=settings.max_faces,
        detection_confidence=settings.detection_confidence,
        tracking_confidence=settings.tracking_confidence,
    )
    try:
        camera = OpenCVCamera(
            settings.camera_index, settings.camera_width, settings.camera_height, settings.camera_fps
        )
    except BaseException:
        detector.close()
        raise

    preview = LandmarkPreview(camera, detector, performance)
    try:
        while True:
            result = preview.next_frame()
            if result is None:
                raise RuntimeError("Camera stopped returning frames")
            metrics = performance.snapshot(result.detector_snapshot)
            cv2.imshow("Face Landmarks", renderer.render(result.frame, result.faces, metrics))
            key = cv2.waitKey(1) & 0xFF
            performance.record_display()
            if key == ord("q"):
                break
    finally:
        try:
            preview.close()
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
