"""PC-only normalization experiment; production preview remains separate."""

import cv2

from face_recognition.application.normalization_experiment import NormalizationExperiment
from face_recognition.application.performance import PerformanceTracker
from face_recognition.application.recognize_face import LandmarkPreview
from face_recognition.config.settings import Settings
from face_recognition.infrastructure.camera.opencv_camera import OpenCVCamera
from face_recognition.infrastructure.features.landmark_normalizer import LandmarkNormalizer
from face_recognition.infrastructure.mediapipe.face_landmarker import MediaPipeFaceLandmarker
from face_recognition.presentation.visualization.frame_renderer import FrameRenderer
from face_recognition.presentation.visualization.normalized_landmarks import NormalizedLandmarkRenderer


def main() -> None:
    run_experiment(Settings())


def run_experiment(settings: Settings) -> None:
    normalizer = LandmarkNormalizer(
        image_width=settings.inference_width, image_height=settings.inference_height,
    )
    experiment = NormalizationExperiment(normalizer, settings.inference_width, settings.inference_height)
    camera_renderer = FrameRenderer("all")
    cloud_renderer = NormalizedLandmarkRenderer()
    performance = PerformanceTracker(settings.metrics_interval_seconds)
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
    try:
        while True:
            frame = preview.next_frame()
            if frame is None:
                raise RuntimeError("Camera stopped returning frames")
            experiment.observe(frame.detector_snapshot.result)
            metrics = performance.snapshot(frame.detector_snapshot)
            cv2.imshow("Normalization experiment - camera", camera_renderer.render(frame.frame, frame.faces, metrics))
            cv2.imshow("Normalization experiment - cloud", cloud_renderer.render(experiment))
            key = cv2.waitKey(1) & 0xFF
            performance.record_display()
            if key == ord("q"):
                break
            if key == ord("r"):
                # Check for a completion that arrived while the GUI pumped keys.
                experiment.observe(detector.snapshot().result)
                experiment.capture_reference()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            cv2.destroyAllWindows()
        finally:
            preview.close()


if __name__ == "__main__":
    main()
