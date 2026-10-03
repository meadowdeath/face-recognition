"""Vendor-independent pixel metadata; pixel storage stays opaque to the domain."""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class PixelFormat(str, Enum):
    BGR = "bgr"
    YUV420_I420 = "yuv420_i420"


@dataclass(frozen=True)
class Frame:
    data: Any
    width: int
    height: int
    pixel_format: PixelFormat

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Frame dimensions must be positive")
        if not isinstance(self.pixel_format, PixelFormat):
            raise ValueError(f"Unsupported pixel format: {self.pixel_format}")
        if self.pixel_format == PixelFormat.YUV420_I420 and (self.width % 2 or self.height % 2):
            raise ValueError("YUV420 frame dimensions must be even")
