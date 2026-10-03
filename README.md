# Face Recognition

Layered Python project for facial landmark detection now and experimental identity recognition later. Development starts on a Windows x86_64 laptop; the intended later target is a Raspberry Pi 3B+ (Linux ARM64/aarch64) with a Raspberry Pi Camera Module 3.

**Current milestone:** laptop webcam → OpenCV capture → MediaPipe Tasks Face Landmarker → landmark visualization with FPS. Press `q` to quit. Face detection and landmarks do not identify a person.

## Stack and architecture

Python 3.13, OpenCV, MediaPipe, NumPy, and scikit-learn; Picamera2 is for Raspberry Pi only.

```text
src/face_recognition/
  domain/          Models and camera, detector, extractor, classifier contracts
  application/     Frame orchestration; dataset and training placeholders
  infrastructure/  OpenCV and optional Picamera2 cameras, MediaPipe detector,
                   feature/classifier/persistence placeholders
  presentation/    CLI and OpenCV frame rendering
  config/          Centralized settings
assets/            Downloaded MediaPipe task model (ignored)
data/raw/          Original samples or metadata (generated contents ignored)
data/processed/    Future vectors and labels (generated contents ignored)
models/            Future trained classifiers (generated contents ignored)
tests/             Hardware-free unit and renderer checks
```

The current application depends on `Camera` and `LandmarkDetector`, not a camera library. A future recognition flow will connect camera → detector → feature extractor → classifier → prediction. The OpenCV GUI stays in presentation.

## Windows laptop setup

From the project root in PowerShell, with Python 3.13 on `PATH`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python tools/download_face_landmarker.py
python -m face_recognition.presentation.cli.recognize
```

The model download uses the [official MediaPipe Face Landmarker bundle](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/index#models). It is stored at `assets/face_landmarker.task` and is not committed. Set camera index, resolution, and detector confidence in `src/face_recognition/config/settings.py`. Run hardware-free checks with `$env:PYTHONPATH='src'; python -m unittest discover -s tests -p 'test_*.py'`.

If `python` resolves to the Windows Store alias, call your Python 3.13 interpreter by its full path. With an existing environment containing the dependencies, set `$env:PYTHONPATH='src'` and run the module command above; a copied environment from another platform is not suitable.

On Raspberry Pi, the planned `Picamera2Camera` adapter will supply BGR frames to the same application and detector. Picamera2 should preferably come from Raspberry Pi OS apt/system packages; it is absent from Windows requirements. Check Python 3.13 and ARM64 package availability on the target before installation. Create a separate environment on each machine: virtual environments contain platform-specific interpreters and binary packages and cannot be copied between Windows and Linux.

## Future work

Dataset capture; experimentally select a smaller landmark subset; normalize translation, scale, and possibly rotation; derive geometric features; compare KNN, SVM, and Random Forest; reject unknown people; integrate the Pi camera; benchmark accuracy and throughput. These are not implemented yet.
