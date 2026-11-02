"""Compose retained camera adapters without importing Raspberry dependencies on laptop."""


def create_live_camera(backend, *, index=0, inference_size=(480, 270), preview_size=(None, None), orientation="normal"):
    if backend == "opencv":
        from face_recognition.infrastructure.camera.opencv_camera import OpenCVCamera
        return OpenCVCamera(index, *preview_size, fps=None, orientation=orientation)
    if backend == "picamera2":
        from face_recognition.config.settings import Settings
        from face_recognition.infrastructure.camera.picamera2_camera import Picamera2Camera
        settings = Settings()
        width, height = preview_size
        return Picamera2Camera(*inference_size, display_width=settings.display_width if width is None else width,
            display_height=settings.display_height if height is None else height, orientation=orientation)
    raise ValueError("Camera backend must be opencv or picamera2")
