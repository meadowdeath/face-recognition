"""Shared lightweight live CLI options; no camera or recognition imports."""

import argparse
from math import isfinite
from pathlib import Path
from face_recognition.config.settings import CAMERA_ORIENTATIONS, Settings


def _positive(value):
    number = int(value)
    if number <= 0: raise argparse.ArgumentTypeError("Dimension must be positive")
    return number


def _timeout(value):
    number = float(value)
    if not isfinite(number) or number <= 0: raise argparse.ArgumentTypeError("Timeout must be finite and positive")
    return number


def add_live_arguments(parser, *, model_required=False):
    settings = Settings()
    parser.add_argument("--camera-backend", choices=("opencv", "picamera2"), default="opencv")
    parser.add_argument("--camera-index", type=int, default=settings.camera_index)
    parser.add_argument("--model-dir", type=Path, required=model_required)
    parser.add_argument("--landmarker-model", type=Path, default=settings.landmarker_model_path)
    parser.add_argument("--inference-width", type=_positive, default=settings.inference_width)
    parser.add_argument("--inference-height", type=_positive, default=settings.inference_height)
    parser.add_argument("--preview-width", type=_positive, help="Optional camera/main-stream width; laptop default is native")
    parser.add_argument("--preview-height", type=_positive, help="Optional camera/main-stream height; laptop default is native")
    parser.add_argument("--orientation", choices=CAMERA_ORIENTATIONS, default=settings.camera_orientation)
    parser.add_argument("--absence-timeout", type=_timeout, default=2.0)
    parser.add_argument("--debug", action="store_true")


def validate_live_arguments(parser, args):
    settings = Settings()
    if args.camera_index < 0: parser.error("--camera-index must be nonnegative")
    if (args.preview_width is None) != (args.preview_height is None): parser.error("Supply both preview dimensions together")
    if args.camera_backend == "picamera2":
        width = settings.display_width if args.preview_width is None else args.preview_width
        height = settings.display_height if args.preview_height is None else args.preview_height
        if (any(v % 2 for v in (width, height, args.inference_width, args.inference_height))
                or args.inference_width > width or args.inference_height > height):
            parser.error("Picamera2 requires even dimensions and inference dimensions no larger than preview")
