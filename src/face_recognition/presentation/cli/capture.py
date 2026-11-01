"""Guided raw-landmark dataset capture with shared asynchronous inference."""

import argparse
from collections.abc import Sequence
from dataclasses import replace
from hashlib import file_digest
from importlib.metadata import version
from pathlib import Path
from typing import cast

from face_recognition.application.capture_dataset import CaptureDataset, CapturePhase, INSTRUCTIONS
from face_recognition.application.performance import PerformanceTracker
from face_recognition.application.recognize_face import LandmarkPreview
from face_recognition.config.settings import CAMERA_ORIENTATIONS, PROJECT_ROOT, Settings
from face_recognition.domain.interfaces.camera import Camera
from face_recognition.domain.interfaces.dataset_repository import DatasetRepository
from face_recognition.domain.models.dataset import CaptureProtocol, CONDITIONS, SessionConfig, validate_identifier
from face_recognition.infrastructure.camera.opencv_camera import OpenCVCamera
from face_recognition.infrastructure.camera.picamera2_camera import Picamera2Camera
from face_recognition.infrastructure.mediapipe.face_landmarker import MediaPipeFaceLandmarker
from face_recognition.infrastructure.persistence.dataset_repository import FilesystemDatasetRepository
from face_recognition.presentation.cli.recognize import _parse_indices, _positive_dimension
from face_recognition.presentation.cli.terminal_controls import TerminalControls
from face_recognition.presentation.visualization.display import DRMDisplay, OpenCVDisplay, OverlayTarget, PreviewDisplay
from face_recognition.presentation.visualization.frame_renderer import FrameRenderer


def _identifier(value: str) -> str:
    try:
        return validate_identifier(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _nonnegative(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Use a nonnegative integer")
    return number


def model_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def print_prompt(capture: CaptureDataset) -> None:
    label = capture.condition.replace("_", " ").upper()
    print(f"\nCONDITION {capture.stage_index + 1}/{len(CONDITIONS)} — {label}\n")
    print(f"({INSTRUCTIONS[capture.stage_index]})\n")
    print("Move into position before SPACE. Allow small natural variation; avoid continuous large movements.")
    print("Press SPACE in the terminal when ready. q or Ctrl+C = cancel.\n", flush=True)


def run_capture(
    settings: Settings, config: SessionConfig, repository: DatasetRepository,
    display_backend: str = "opencv", renderer: FrameRenderer | None = None,
) -> None:
    if display_backend not in ("opencv", "drm", "none"):
        raise ValueError("Unknown display backend")
    if display_backend == "drm" and config.camera_backend != "picamera2":
        raise ValueError("DRM display requires Picamera2")
    capture = CaptureDataset(repository, config)  # Reject incompatible sessions before opening hardware.
    if capture.phase is CapturePhase.DONE:
        print("Session already complete; no samples overwritten.")
        return
    renderer = renderer if renderer is not None else FrameRenderer("all")
    protocol = config.protocol
    with TerminalControls() as controls:
        performance = PerformanceTracker(settings.metrics_interval_seconds)
        detector = MediaPipeFaceLandmarker(
            model_path=settings.landmarker_model_path, max_faces=1,
            detection_confidence=config.detection_confidence, tracking_confidence=config.tracking_confidence,
            inference_size=(config.inference_width, config.inference_height),
        )
        try:
            camera: Camera
            if config.camera_backend == "opencv":
                camera = OpenCVCamera(int(config.camera_device_id), config.display_width, config.display_height,
                                      config.requested_camera_fps or settings.camera_fps, orientation=config.orientation)
            else:
                camera = Picamera2Camera(config.inference_width, config.inference_height,
                                        display_width=config.display_width, display_height=config.display_height,
                                        orientation=config.orientation)
        except BaseException:
            detector.close()
            raise
        preview = LandmarkPreview(camera, detector, performance)
        display: PreviewDisplay | None = None
        progress_width = 0
        previous_progress = ""
        try:
            if display_backend == "opencv":
                display = OpenCVDisplay(renderer)
            elif display_backend == "drm":
                display = DRMDisplay(cast(OverlayTarget, camera), renderer, config.display_width, config.display_height,
                                     interval_seconds=settings.metrics_interval_seconds)
            # No-display mode deliberately has no benchmark reporter competing with guided progress.
            print(f"Raw landmark capture: {config.person_id}/{config.session_id}; "
                  f"{len(capture.manifest.completed_conditions)}/{len(CONDITIONS)} conditions already saved.")
            print(f"Protocol: {protocol.warmup_results} warm-up results, {protocol.samples_per_condition} samples, "
                  f">= {protocol.minimum_sample_interval_ms} ms spacing. Keep only the intended participant in view.")
            print_prompt(capture)
            while capture.phase is not CapturePhase.DONE:
                waiting_for_space = capture.phase is CapturePhase.WAITING
                frame = preview.next_frame()
                if frame is None:
                    raise RuntimeError("Camera stopped returning frames")
                stage_index = capture.stage_index
                changed = capture.observe(frame.detector_snapshot.result)
                if capture.stage_index != stage_index:
                    print(f"\nSaved {protocol.samples_per_condition} samples for {CONDITIONS[stage_index]}.", flush=True)
                    progress_width = 0
                    previous_progress = ""
                    if capture.phase is CapturePhase.DONE:
                        print(f"Session complete: {len(CONDITIONS) * protocol.samples_per_condition} raw samples.")
                        break
                    print_prompt(capture)
                elif changed:
                    progress = (f"Warm-up: {capture.warmup_count}/{protocol.warmup_results}  "
                                f"Collecting: {capture.accepted_count}/{protocol.samples_per_condition}")
                    if capture.status:
                        progress += "  " + capture.status
                    if progress != previous_progress:
                        progress_width = max(progress_width, len(progress))
                        print("\r" + progress.ljust(progress_width), end="", flush=True)
                        previous_progress = progress
                if display is not None:
                    keep_running = display.show(frame, performance.snapshot(frame.detector_snapshot))
                    if display_backend == "opencv":
                        performance.record_display()
                    elif cast(DRMDisplay, display).overlay_updated:
                        performance.record_overlay_update()
                    if not keep_running:
                        capture.cancel()
                        print("\nCapture cancelled; completed conditions retained, unfinished batch discarded.")
                        break
                key = controls.poll()
                if key == "q":
                    capture.cancel()
                    print("\nCapture cancelled; completed conditions retained, unfinished batch discarded.")
                    break
                if waiting_for_space and key == " ":
                    capture.observe(detector.snapshot().result)  # Exclude a completion already available at SPACE.
                    capture.start_condition()
        except KeyboardInterrupt:
            capture.cancel()
            print("\nCapture interrupted; completed conditions retained, unfinished batch discarded.")
        finally:
            try:
                if display is not None:
                    display.close()
            finally:
                preview.close()


def main(argv: Sequence[str] | None = None) -> None:
    settings = Settings()
    parser = argparse.ArgumentParser(description="Guided raw MediaPipe landmark dataset capture")
    parser.add_argument("--person-id", required=True, type=_identifier, help="Pseudonymous identifier, e.g. p001")
    parser.add_argument("--session-id", required=True, type=_identifier, help="Independent session identifier, e.g. session_01")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--camera", choices=("opencv", "picamera2"), default="opencv")
    parser.add_argument("--camera-index", type=_nonnegative, default=settings.camera_index)
    parser.add_argument("--display", choices=("opencv", "drm", "none"), default="opencv")
    parser.add_argument("--orientation", choices=CAMERA_ORIENTATIONS, default=settings.camera_orientation)
    for name in ("inference_width", "inference_height", "display_width", "display_height"):
        parser.add_argument("--" + name.replace("_", "-"), type=_positive_dimension, default=getattr(settings, name))
    protocol = CaptureProtocol()
    parser.add_argument("--warmup-results", type=_nonnegative, default=protocol.warmup_results)
    parser.add_argument("--samples-per-condition", type=_positive_dimension, default=protocol.samples_per_condition)
    parser.add_argument("--minimum-sample-interval-ms", type=_nonnegative, default=protocol.minimum_sample_interval_ms)
    parser.add_argument("--landmarks", choices=FrameRenderer.MODES, default="all")
    parser.add_argument("--indices", type=_parse_indices, default=())
    args = parser.parse_args(argv)
    if args.display == "drm" and args.camera != "picamera2":
        parser.error("--display drm requires --camera picamera2")
    if args.landmarks == "selected" and not args.indices:
        parser.error("--landmarks selected requires --indices (display only)")
    dimensions = (args.inference_width, args.inference_height, args.display_width, args.display_height)
    if args.camera == "picamera2":
        if args.camera_index != 0:
            parser.error("The existing Picamera2 adapter uses its default device 0")
        if any(value % 2 for value in dimensions):
            parser.error("Picamera2 dimensions must be even")
        if args.inference_width > args.display_width or args.inference_height > args.display_height:
            parser.error("Picamera2 inference dimensions must not exceed display dimensions")
    settings = replace(settings, camera_index=args.camera_index, camera_orientation=args.orientation,
                       **{name: getattr(args, name) for name in ("inference_width", "inference_height", "display_width", "display_height")})
    try:
        config = SessionConfig(
            args.person_id, args.session_id, args.camera, str(args.camera_index), args.orientation,
            args.inference_width, args.inference_height, args.display_width, args.display_height,
            model_digest(settings.landmarker_model_path), version("mediapipe"),
            settings.detection_confidence, settings.tracking_confidence,
            settings.camera_fps if args.camera == "opencv" else None,
            CaptureProtocol(args.warmup_results, args.samples_per_condition, args.minimum_sample_interval_ms),
        )
        run_capture(settings, config, FilesystemDatasetRepository(args.data_root), args.display,
                    FrameRenderer(args.landmarks, args.indices))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Capture error: {exc}\n")


if __name__ == "__main__":
    main()
