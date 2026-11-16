"""One shared V3 live recognition path with isolated OpenCV/Picamera2 acquisition."""

import argparse
from pathlib import Path
import sys
from time import monotonic

from face_recognition.application.live_recognition import LiveOpenSetRecognition
from face_recognition.application.performance import PerformanceTracker
from face_recognition.application.recognize_face import LandmarkPreview
from face_recognition.config.settings import Settings
from face_recognition.presentation.cli.live_options import add_live_arguments, validate_live_arguments
from face_recognition.domain.models.rejection import RejectionError
from face_recognition.infrastructure.camera.live_camera import create_live_camera
from face_recognition.infrastructure.features.landmark_feature_extractor import LandmarkFeatureExtractor
from face_recognition.infrastructure.features.pose_landmark_normalizer import PoseLandmarkNormalizer
from face_recognition.infrastructure.mediapipe.face_landmarker import MediaPipeFaceLandmarker
from face_recognition.infrastructure.persistence.recognition_model import load_recognition_bundle, RuntimeCompatibilityError
from face_recognition.presentation.visualization.display import DRMDisplay, OpenCVDisplay
from face_recognition.presentation.visualization.live_open_set_renderer import LiveOpenSetRenderer


def run_live_recognition(model_dir, *, camera_backend="opencv", camera_index=0, display_backend="auto", debug=False,
                         landmarker_model=None, inference_size=(480, 270), preview_size=(None, None),
                         orientation="normal", absence_timeout=2.0):
    settings = Settings()
    if camera_backend not in ("opencv", "picamera2") or display_backend not in ("auto", "opencv", "drm"):
        raise ValueError("Unsupported camera/display backend")
    display_backend = ("drm" if camera_backend == "picamera2" else "opencv") if display_backend == "auto" else display_backend
    if display_backend == "drm" and camera_backend != "picamera2":
        raise ValueError("DRM display requires Picamera2")
    # Integrity/runtime validation happens once and before any camera opens.
    recognizer = load_recognition_bundle(Path(model_dir))
    pipeline = LiveOpenSetRecognition(recognizer, PoseLandmarkNormalizer(*inference_size), LandmarkFeatureExtractor(),
        absence_timeout=absence_timeout, metrics_interval=settings.metrics_interval_seconds, clock=monotonic, profile=debug)
    renderer = LiveOpenSetRenderer(debug)
    performance = PerformanceTracker(settings.metrics_interval_seconds, clock=monotonic)
    detector = MediaPipeFaceLandmarker(model_path=settings.landmarker_model_path if landmarker_model is None else Path(landmarker_model),
        max_faces=1, detection_confidence=settings.detection_confidence, tracking_confidence=settings.tracking_confidence,
        inference_size=inference_size)
    try:
        camera = create_live_camera(camera_backend, index=camera_index, inference_size=inference_size,
            preview_size=preview_size, orientation=orientation)
    except BaseException:
        detector.close()
        raise
    preview = LandmarkPreview(camera, detector, performance)
    display = None
    try:
        if display_backend == "opencv":
            display = OpenCVDisplay(renderer, window_title="V3 Open-set Recognition", exit_keys=(ord("q"), 27))
        else:
            width = settings.display_width if preview_size[0] is None else preview_size[0]
            height = settings.display_height if preview_size[1] is None else preview_size[1]
            display = DRMDisplay(camera, renderer, width, height, interval_seconds=settings.metrics_interval_seconds, clock=monotonic)
        while True:
            frame = preview.next_frame()
            if frame is None:
                raise RuntimeError("Camera stopped returning frames")
            view = pipeline.process(frame.detector_snapshot)
            renderer.update(view, pipeline.metrics())
            metrics = performance.snapshot(frame.detector_snapshot)
            keep_running = display.show(frame, metrics)
            if display_backend == "opencv": performance.record_display()
            elif display.overlay_updated: performance.record_overlay_update()
            if not keep_running: break
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if display is not None: display.close()
        finally:
            preview.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Live frozen V3 single-face open-set recognition; no training or calibration")
    add_live_arguments(parser, model_required=True)
    parser.add_argument("--display", choices=("auto", "opencv", "drm"), default="auto")
    args = parser.parse_args(argv)
    validate_live_arguments(parser, args)
    if args.display == "drm" and args.camera_backend != "picamera2": parser.error("DRM requires Picamera2")
    try:
        run_live_recognition(args.model_dir, camera_backend=args.camera_backend, camera_index=args.camera_index,
            display_backend=args.display, debug=args.debug, landmarker_model=args.landmarker_model,
            inference_size=(args.inference_width, args.inference_height), preview_size=(args.preview_width, args.preview_height),
            orientation=args.orientation, absence_timeout=args.absence_timeout)
    except KeyboardInterrupt:
        return 0
    except (RuntimeCompatibilityError, RejectionError, FileNotFoundError) as exc:
        print(f"Live recognition stopped: {exc}", file=sys.stderr)
        return 1
    except Exception:
        print("Live recognition stopped: camera, display or inference failed. Check the selected backend, GUI access and model assets.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
