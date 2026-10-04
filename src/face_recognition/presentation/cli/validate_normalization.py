"""Terminal-guided PC experiment; SPACE starts each static condition."""

import os

import cv2

from face_recognition.application.normalization_validation import (
    CONDITIONS, STAGES, ConditionReport, NormalizationValidation, Phase,
    RMSEStatistics, ValidationConfig,
)
from face_recognition.application.performance import PerformanceTracker
from face_recognition.application.recognize_face import LandmarkPreview
from face_recognition.config.settings import Settings
from face_recognition.infrastructure.camera.opencv_camera import OpenCVCamera
from face_recognition.infrastructure.features.landmark_normalizer import LandmarkNormalizer
from face_recognition.infrastructure.mediapipe.face_landmarker import MediaPipeFaceLandmarker
from face_recognition.presentation.visualization.frame_renderer import FrameRenderer


def terminal_key() -> str | None:
    """Poll Windows console without blocking capture or requiring Enter.

    The camera window also accepts SPACE/q; non-Windows PCs use that fallback.
    """
    if os.name != "nt":
        return None
    import msvcrt

    if not msvcrt.kbhit():
        return None
    key = msvcrt.getwch()
    if key in ("\x00", "\xe0"):
        if msvcrt.kbhit():
            msvcrt.getwch()  # Discard extended-key code, e.g. arrow keys.
        return None
    if key == "\x03":
        raise KeyboardInterrupt
    return key.lower()


def stage_title(index: int) -> str:
    return "REFERENCE" if index == 0 else f"TEST {index}/{len(CONDITIONS)} — {STAGES[index].name}"


def print_prompt(session: NormalizationValidation) -> None:
    print(f"\n{stage_title(session.stage_index)}\n")
    print(f"({session.condition.description})\n")
    print("Finish moving before pressing SPACE, then remain still during collection.")
    print("Press SPACE when ready. q or Ctrl+C = quit.\n", flush=True)


def print_statistics(name: str, stats: RMSEStatistics) -> None:
    print(f"\n{name}")
    for label, value in (("mean", stats.mean), ("std", stats.std), ("median", stats.median),
                         ("min", stats.minimum), ("max", stats.maximum)):
        print(f"  {label:6s}: {value:.6f}")


def print_report(index: int, report: ConditionReport) -> None:
    print(f"\n{stage_title(index)}\nSamples: {report.samples}")
    print_statistics("Raw RMSE (image-width units)", report.raw)
    print_statistics("Normalized RMSE (interocular units)", report.normalized)
    print("\nCompleted.", flush=True)


def print_summary(session: NormalizationValidation) -> None:
    print("\nNORMALIZATION VALIDATION SUMMARY\n")
    print("Population std; raw and normalized columns use different coordinate scales.")
    print(f"{'Condition':20s} {'Raw mean':>10s} {'Raw std':>10s} {'Raw median':>10s} "
          f"{'Norm mean':>10s} {'Norm std':>10s} {'Norm median':>11s}")
    for report in session.reports:
        print(f"{report.condition.name:20s} {report.raw.mean:10.6f} {report.raw.std:10.6f} "
              f"{report.raw.median:10.6f} {report.normalized.mean:10.6f} "
              f"{report.normalized.std:10.6f} {report.normalized.median:11.6f}")
    print("\nNo pass/fail thresholds. Nothing saved to disk.", flush=True)


def run_validation(settings: Settings, config: ValidationConfig = ValidationConfig()) -> None:
    session = NormalizationValidation(
        LandmarkNormalizer(image_width=settings.inference_width, image_height=settings.inference_height),
        settings.inference_width, settings.inference_height, config,
    )
    performance = PerformanceTracker(settings.metrics_interval_seconds)
    renderer = FrameRenderer("all")
    detector = MediaPipeFaceLandmarker(
        model_path=settings.landmarker_model_path, max_faces=settings.max_faces,
        detection_confidence=settings.detection_confidence,
        tracking_confidence=settings.tracking_confidence,
        inference_size=(settings.inference_width, settings.inference_height),
    )
    try:
        camera = OpenCVCamera(settings.camera_index, settings.display_width, settings.display_height,
                              settings.camera_fps, orientation=settings.camera_orientation)
    except BaseException:
        detector.close()
        raise
    preview = LandmarkPreview(camera, detector, performance)
    progress_width = 0
    previous_error = None
    try:
        print("Guided static-pose validation; keep the same subject throughout.")
        print(f"Each stage: {config.warmup_results} warm-up completions, "
              f"{config.measurement_results} valid measurement results.")
        print("Statistics use population std. Raw and normalized RMSE have different units.")
        print("Use SPACE/q in PowerShell (no Enter needed) or in the camera window.")
        print_prompt(session)
        while session.phase is not Phase.DONE:
            waiting_for_space = session.phase is Phase.WAITING
            frame = preview.next_frame()
            if frame is None:
                raise RuntimeError("Camera stopped returning frames")
            stage_index = session.stage_index
            changed = session.observe(frame.detector_snapshot.result)
            if session.stage_index != stage_index:
                print()
                progress_width = 0
                previous_error = None
                if stage_index == 0:
                    print(f"Reference templates built from {config.measurement_results} samples.")
                    print("CENTER will use a separate, independent measurement set.\nCompleted.")
                else:
                    print_report(stage_index, session.reports[-1])
                if session.phase is Phase.DONE:
                    print_summary(session)
                    break
                print_prompt(session)
            elif changed:
                if session.error is not None and session.error != previous_error:
                    print(f"\n{session.error}")
                    progress_width = 0
                previous_error = session.error
                progress = (f"Warm-up: {session.warmup_count}/{config.warmup_results}  "
                            f"Collecting: {session.measurement_count}/{config.measurement_results}")
                progress_width = max(progress_width, len(progress))
                print("\r" + progress.ljust(progress_width), end="", flush=True)
            metrics = performance.snapshot(frame.detector_snapshot)
            cv2.imshow("Guided normalization validation - camera", renderer.render(frame.frame, frame.faces, metrics))
            window_key = cv2.waitKey(1) & 0xFF
            performance.record_display()
            console_key = terminal_key()
            if console_key == "q" or window_key == ord("q"):
                print("\nValidation stopped; in-memory results discarded.")
                break
            if waiting_for_space and (console_key == " " or window_key == ord(" ")):
                # Exclude a completion that arrived during the GUI/key poll.
                session.observe(detector.snapshot().result)
                session.start_stage()
    except KeyboardInterrupt:
        print("\nValidation interrupted; in-memory results discarded.")
    finally:
        try:
            cv2.destroyAllWindows()
        finally:
            preview.close()


def main() -> None:
    run_validation(Settings())


if __name__ == "__main__":
    main()
