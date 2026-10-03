# Face Recognition

Layered Python project for facial landmark detection now and experimental identity recognition later. Development starts on a Windows x86_64 laptop; the intended later target is a Raspberry Pi 3B+ (Linux ARM64/aarch64) with a Raspberry Pi Camera Module 3.

**Current milestone:** OpenCV webcam or Picamera2 camera → asynchronous MediaPipe Tasks Face Landmarker → landmark visualization with independent performance metrics. Press `q` to quit. Face detection and landmarks do not identify a person.

## Stack and architecture

**Python 3.11 is the only currently supported Python version** (`>=3.11,<3.12`). The mandatory runtime dependencies are NumPy 1.26.4, opencv-contrib-python 4.10.0.84, MediaPipe 0.10.14, and protobuf 4.25.9. The same versions are pinned in `pyproject.toml` and `requirements.txt`; `requirements-rpi.txt` includes those shared requirements.

The reference Raspberry Pi environment is a Raspberry Pi 3B+ running Debian 12 Bookworm on aarch64 with Python 3.11.2. It supplies Picamera2 0.3.31, libcamera, and python3-libcamera through system packages. Windows uses the shared Python dependencies without Picamera2. Scikit-learn is not required for this landmark-preview milestone.

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

The current application depends on `Camera` and `LandmarkDetector`, not a camera library. The CLI selects `--camera opencv` (default) or `--camera picamera2`; both adapters use the same preview loop. The OpenCV GUI stays in presentation.

## Asynchronous preview

`LandmarkDetector` exposes `submit(frame)` and `snapshot()`. `LandmarkPreview.next_frame()` reads a camera frame, submits it, and immediately retrieves the latest completed result. It never waits for inference to finish; startup frames may have no landmarks, and several frames may reuse a result. Landmarks can therefore lag a moving face.

The MediaPipe adapter uses Tasks `RunningMode.LIVE_STREAM` and `detect_async()`. Its callback updates only one latest-result slot, a timestamp, and a completion count. Conversion to domain landmarks happens on the main loop once per new result. MediaPipe may drop submissions when busy; the project creates no frame queue or custom inference worker. Optional blendshapes and facial transformation matrices are disabled, with one face configured by default.

Capture and display run on the main loop. The OpenCV backend requests 30 FPS; achieved capture/display rates depend on the device and processing overhead. About 30 FPS capture/display and at least 15 completed inferences per second on the Pi are goals, not verified Pi performance.

The overlay reports:

- **Capture FPS:** successful camera reads per second.
- **Display FPS:** completed preview iterations (`imshow` plus the GUI event pump) per second; this does not measure monitor refresh.
- **Inference FPS:** completed callback results per second, including results with no face.
- **Submitted / completed:** cumulative successful `detect_async()` calls and callback completions. Submitted frames may be dropped internally; their difference also includes work still in flight.
- **Result latency:** approximate submission-to-callback milliseconds for the latest result, including input conversion and dispatch. It excludes camera acquisition, rendering, and the age of a reused result.

Rates update over measured intervals of at least one second and initially show zero. Settings centralize the requested camera FPS, metrics interval, resolution, confidences, model paths, and renderer defaults. Shutdown stops the detector callbacks, releases the camera, and destroys OpenCV windows, including on pipeline errors.

## Windows laptop setup

From the project root in PowerShell, with Python 3.11 on `PATH`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python tools/download_face_landmarker.py
python -m face_recognition.presentation.cli.recognize --camera opencv
```

Benchmark rendering with `--landmarks none`, `--landmarks all` (the default), or `--landmarks selected --indices INDEX,INDEX,...`. Selected mode requires your explicit nonnegative landmark indices; out-of-range indices are skipped. No recognition subset is predefined. Every mode displays the same metrics. Rendering fewer points measures rendering/post-processing overhead; it does **not** reduce MediaPipe neural-network inference cost.

The model download uses the [official MediaPipe Face Landmarker bundle](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/index#models). It is stored at `assets/face_landmarker.task` and is not committed. Set camera index, resolution, and detector confidence in `src/face_recognition/config/settings.py`. Run hardware-free checks with `$env:PYTHONPATH='src'; python -m unittest discover -s tests -p 'test_*.py'`.

If `python` resolves to the Windows Store alias, call your Python 3.11 interpreter by its full path. With the existing Windows Conda environment, activate `face-recognition`, set `$env:PYTHONPATH='src'`, and run the module command above. Install the pinned runtime dependencies with `python -m pip install -r requirements.txt` if needed.

## Raspberry Pi environment

Ensure Picamera2 0.3.31, libcamera, and python3-libcamera are available through Debian/Raspberry Pi OS apt/system packages. These are prerequisites, not pip requirements. With the system Python 3.11 interpreter, create the environment from the project root:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements-rpi.txt
python tools/download_face_landmarker.py
PYTHONPATH=src python -m face_recognition.presentation.cli.recognize --camera picamera2
```

`--system-site-packages` is required so the environment can access the system-provided Picamera2 and libcamera Python bindings. Run the command above from the project root in a graphical desktop session for the OpenCV preview window. The `Picamera2Camera` adapter uses the configured width and height and imports Picamera2 only when instantiated on Linux. This backend selection is covered by hardware-free tests; execution and performance on Pi hardware still require validation.

Create a separate environment on each machine: virtual environments contain platform-specific interpreters and binary packages and cannot be copied between Windows and Linux. Share source, configuration, compatible model assets, and the pinned requirements.

## Future work

Dataset capture; experimentally select a smaller landmark subset; normalize translation, scale, and possibly rotation; derive geometric features; compare KNN, SVM, and Random Forest; reject unknown people; validate the Pi camera on hardware; benchmark accuracy and throughput. These are not implemented yet.
