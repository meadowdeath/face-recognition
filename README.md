# Face Recognition

Layered Python project for facial landmark detection now and experimental identity recognition later. Development starts on a Windows x86_64 laptop; the intended later target is a Raspberry Pi 3B+ (Linux ARM64/aarch64) with a Raspberry Pi Camera Module 3.

**Current milestone:** laptop webcam → OpenCV capture → asynchronous MediaPipe Tasks Face Landmarker → landmark visualization with independent performance metrics. Press `q` to quit. Face detection and landmarks do not identify a person.

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

The current application depends on `Camera` and `LandmarkDetector`, not a camera library. The OpenCV GUI stays in presentation.

## Asynchronous preview

`LandmarkDetector` exposes `submit(frame)` and `snapshot()`. `LandmarkPreview.next_frame()` reads a camera frame, submits it, and immediately retrieves the latest completed result. It never waits for inference to finish; startup frames may have no landmarks, and several frames may reuse a result. Landmarks can therefore lag a moving face.

The MediaPipe adapter uses Tasks `RunningMode.LIVE_STREAM` and `detect_async()`. Its callback updates only one latest-result slot, a timestamp, and a completion count. Conversion to domain landmarks happens on the main loop once per new result. MediaPipe may drop submissions when busy; the project creates no frame queue or custom inference worker. Optional blendshapes and facial transformation matrices are disabled, with one face configured by default.

Capture and display run on the main loop. The camera is requested to run at 30 FPS; achieved capture/display rates depend on the device and processing overhead. About 30 FPS capture/display and at least 15 completed inferences per second on the eventual Pi are goals, not verified Pi performance.

The overlay reports:

- **Capture FPS:** successful camera reads per second.
- **Display FPS:** completed preview iterations (`imshow` plus the GUI event pump) per second; this does not measure monitor refresh.
- **Inference FPS:** completed callback results per second, including results with no face.
- **Submitted / completed:** cumulative successful `detect_async()` calls and callback completions. Submitted frames may be dropped internally; their difference also includes work still in flight.
- **Result latency:** approximate submission-to-callback milliseconds for the latest result, including input conversion and dispatch. It excludes camera acquisition, rendering, and the age of a reused result.

Rates update over measured intervals of at least one second and initially show zero. Settings centralize the requested camera FPS, metrics interval, resolution, confidences, model paths, and renderer defaults. Shutdown stops the detector callbacks, releases the camera, and destroys OpenCV windows, including on pipeline errors.

## Windows laptop setup

From the project root in PowerShell, with Python 3.13 on `PATH`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python tools/download_face_landmarker.py
python -m face_recognition.presentation.cli.recognize
```

Benchmark rendering with `--landmarks none`, `--landmarks all` (the default), or `--landmarks selected --indices INDEX,INDEX,...`. Selected mode requires your explicit nonnegative landmark indices; out-of-range indices are skipped. No recognition subset is predefined. Every mode displays the same metrics. Rendering fewer points measures rendering/post-processing overhead; it does **not** reduce MediaPipe neural-network inference cost.

The model download uses the [official MediaPipe Face Landmarker bundle](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/index#models). It is stored at `assets/face_landmarker.task` and is not committed. Set camera index, resolution, and detector confidence in `src/face_recognition/config/settings.py`. Run hardware-free checks with `$env:PYTHONPATH='src'; python -m unittest discover -s tests -p 'test_*.py'`.

If `python` resolves to the Windows Store alias, call your Python 3.13 interpreter by its full path. With an existing environment containing the dependencies, set `$env:PYTHONPATH='src'` and run the module command above; a copied environment from another platform is not suitable.

On Raspberry Pi, the planned `Picamera2Camera` adapter will supply BGR frames to the same application and asynchronous detector contract. Picamera2 should preferably come from Raspberry Pi OS apt/system packages; it is absent from Windows requirements. Check Python 3.13 and ARM64 package availability on the target before installation. Create a separate environment on each machine: virtual environments contain platform-specific interpreters and binary packages and cannot be copied between Windows and Linux.

## Future work

Dataset capture; experimentally select a smaller landmark subset; normalize translation, scale, and possibly rotation; derive geometric features; compare KNN, SVM, and Random Forest; reject unknown people; integrate the Pi camera; benchmark accuracy and throughput. These are not implemented yet.
