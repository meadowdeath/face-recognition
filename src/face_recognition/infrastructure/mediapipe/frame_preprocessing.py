"""Shared RGB preparation, called only for frames selected for inference."""

import cv2
import numpy as np

from face_recognition.domain.models.frame import Frame, PixelFormat


def prepare_rgb(frame: Frame | np.ndarray, inference_size: tuple[int, int] | None) -> np.ndarray:
    if isinstance(frame, Frame) and frame.pixel_format == PixelFormat.YUV420_I420:
        rgb = cv2.cvtColor(frame.data, cv2.COLOR_YUV2RGB_I420)
        rgb = rgb[:frame.height, :frame.width]
        if inference_size is not None and (frame.width, frame.height) != inference_size:
            rgb = cv2.resize(rgb, inference_size, interpolation=cv2.INTER_AREA)
        return rgb
    bgr = frame.data if isinstance(frame, Frame) else frame
    if inference_size is not None and (bgr.shape[1], bgr.shape[0]) != inference_size:
        bgr = cv2.resize(bgr, inference_size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
