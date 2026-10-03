"""Download Google's Face Landmarker task model into the local assets directory."""

from pathlib import Path
from urllib.request import urlopen

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
DESTINATION = Path(__file__).resolve().parents[1] / "assets" / "face_landmarker.task"


def main() -> None:
    if DESTINATION.is_file():
        print(f"Model already present: {DESTINATION}")
        return
    temporary = DESTINATION.with_suffix(".task.tmp")
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(MODEL_URL, timeout=60) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if temporary.stat().st_size == 0:
            raise RuntimeError("Downloaded model is empty")
        temporary.replace(DESTINATION)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Downloaded: {DESTINATION}")


if __name__ == "__main__":
    main()
