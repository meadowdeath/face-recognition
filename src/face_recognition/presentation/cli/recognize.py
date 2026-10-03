"""Laptop webcam landmark preview; identity recognition is not implemented."""

from time import perf_counter

import cv2

from face_recognition.application.recognize_face import LandmarkPreview
from face_recognition.config.settings import Settings
from face_recognition.infrastructure.camera.opencv_camera import OpenCVCamera
from face_recognition.infrastructure.mediapipe.face_landmarker import MediaPipeFaceLandmarker
from face_recognition.presentation.visualization.frame_renderer import FrameRenderer


def main() -> None:
    settings = Settings()
    detector = MediaPipeFaceLandmarker(
        model_path=settings.landmarker_model_path,
        max_faces=settings.max_faces,
        detection_confidence=settings.detection_confidence,
        tracking_confidence=settings.tracking_confidence,
    )
    try:
        camera = OpenCVCamera(settings.camera_index, settings.camera_width, settings.camera_height)
    except BaseException:
        detector.close()
        raise

    preview = LandmarkPreview(camera, detector)
    renderer = FrameRenderer()
    previous = perf_counter()
    try:
        while True:
            result = preview.next_frame()
            if result is None:
                raise RuntimeError("Camera stopped returning frames")
            current = perf_counter()
            fps = 1.0 / max(current - previous, 1e-9)
            previous = current
            cv2.imshow("Face Landmarks", renderer.render(result.frame, result.faces, fps))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        try:
            preview.close()
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
