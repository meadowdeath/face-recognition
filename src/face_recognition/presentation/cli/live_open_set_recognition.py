"""One shared V3 live recognition path with isolated OpenCV/Picamera2 acquisition."""

import argparse
from math import isfinite
from pathlib import Path
import sys
from time import monotonic

from face_recognition.application.live_open_set_recognition import LiveOpenSetRecognition
from face_recognition.application.performance import PerformanceTracker
from face_recognition.application.recognize_face import LandmarkPreview
from face_recognition.config.settings import CAMERA_ORIENTATIONS, Settings
from face_recognition.domain.models.rejection import RejectionError
from face_recognition.infrastructure.camera.live_camera import create_live_camera
from face_recognition.infrastructure.features.landmark_feature_extractor import LandmarkFeatureExtractor
from face_recognition.infrastructure.features.pose_landmark_normalizer import PoseLandmarkNormalizer
from face_recognition.infrastructure.mediapipe.face_landmarker import MediaPipeFaceLandmarker
from face_recognition.infrastructure.persistence.v3_deployable_model import load_v3_deployable_bundle, V3RuntimeCompatibilityError
from face_recognition.presentation.visualization.display import DRMDisplay, OpenCVDisplay
from face_recognition.presentation.visualization.live_open_set_renderer import LiveOpenSetRenderer


def _positive(value):
    number = int(value)
    if number <= 0: raise argparse.ArgumentTypeError("Dimension must be positive")
    return number


def _timeout(value):
    number = float(value)
    if not isfinite(number) or number <= 0: raise argparse.ArgumentTypeError("Timeout must be finite and positive")
    return number


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
    recognizer = load_v3_deployable_bundle(Path(model_dir))
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
    settings = Settings()
    parser = argparse.ArgumentParser(description="Live frozen V3 single-face open-set recognition; no training or calibration")
    parser.add_argument("--camera-backend", choices=("opencv", "picamera2"), default="opencv")
    parser.add_argument("--camera-index", type=int, default=settings.camera_index)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--landmarker-model", type=Path, default=settings.landmarker_model_path)
    parser.add_argument("--display", choices=("auto", "opencv", "drm"), default="auto")
    parser.add_argument("--inference-width", type=_positive, default=settings.inference_width)
    parser.add_argument("--inference-height", type=_positive, default=settings.inference_height)
    parser.add_argument("--preview-width", type=_positive, help="Optional camera/main-stream width; laptop default is native")
    parser.add_argument("--preview-height", type=_positive, help="Optional camera/main-stream height; laptop default is native")
    parser.add_argument("--orientation", choices=CAMERA_ORIENTATIONS, default=settings.camera_orientation)
    parser.add_argument("--absence-timeout", type=_timeout, default=2.0)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)
    if args.camera_index < 0: parser.error("--camera-index must be nonnegative")
    if (args.preview_width is None) != (args.preview_height is None): parser.error("Supply both preview dimensions together")
    if args.display == "drm" and args.camera_backend != "picamera2": parser.error("DRM requires Picamera2")
    if args.camera_backend == "picamera2":
        width = settings.display_width if args.preview_width is None else args.preview_width
        height = settings.display_height if args.preview_height is None else args.preview_height
        if (any(v % 2 for v in (width, height, args.inference_width, args.inference_height))
                or args.inference_width > width or args.inference_height > height):
            parser.error("Picamera2 requires even dimensions and inference dimensions no larger than preview")
    try:
        run_live_recognition(args.model_dir, camera_backend=args.camera_backend, camera_index=args.camera_index,
            display_backend=args.display, debug=args.debug, landmarker_model=args.landmarker_model,
            inference_size=(args.inference_width, args.inference_height), preview_size=(args.preview_width, args.preview_height),
            orientation=args.orientation, absence_timeout=args.absence_timeout)
    except KeyboardInterrupt:
        return 0
    except (V3RuntimeCompatibilityError, RejectionError, FileNotFoundError) as exc:
        print(f"Live recognition stopped: {exc}", file=sys.stderr)
        return 1
    except Exception:
        print("Live recognition stopped: camera, display or inference failed. Check the selected backend, GUI access and model assets.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
