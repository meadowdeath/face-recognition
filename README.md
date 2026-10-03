# Face Recognition

Layered Python project for facial landmark detection now and experimental identity recognition later. Development starts on a Windows x86_64 laptop; the intended later target is a Raspberry Pi 3B+ (Linux ARM64/aarch64) with a Raspberry Pi Camera Module 3.

**Current milestone:** OpenCV webcam or Picamera2 camera → asynchronous MediaPipe Tasks Face Landmarker → OpenCV window, native DRM preview, or console benchmark metrics. Press `q` in the OpenCV window or Ctrl+C to quit. Face detection and landmarks do not identify a person.

## Stack and architecture

**Python 3.11 is the only currently supported Python version** (`>=3.11,<3.12`). The mandatory runtime dependencies are NumPy 1.26.4, opencv-contrib-python 4.10.0.84, MediaPipe 0.10.14, and protobuf 4.25.9. The same versions are pinned in `pyproject.toml` and `requirements.txt`; `requirements-rpi.txt` includes those shared requirements.

The reference Raspberry Pi environment is a Raspberry Pi 3B+ running Debian 12 Bookworm on aarch64 with Python 3.11.2. It supplies Picamera2 0.3.31, libcamera, and python3-libcamera through system packages. Windows uses the shared Python dependencies without Picamera2. Scikit-learn is not required for this landmark-preview milestone.

```text
src/face_recognition/
  domain/          Models and camera, detector, extractor, classifier contracts
  application/     Frame orchestration; dataset and training placeholders
  infrastructure/  OpenCV and optional Picamera2 cameras, MediaPipe detector,
                   feature/classifier/persistence placeholders
  presentation/    CLI, frame/overlay rendering, and selectable display adapters
  config/          Centralized settings
assets/            Downloaded MediaPipe task model (ignored)
data/raw/          Original samples or metadata (generated contents ignored)
data/processed/    Future vectors and labels (generated contents ignored)
models/            Future trained classifiers (generated contents ignored)
tests/             Hardware-free unit and renderer checks
```

The current application depends on `Camera` and `LandmarkDetector`, not a camera library. The CLI selects `--camera opencv` (default) or `--camera picamera2`, plus `--display opencv` (default), `--display drm`, or `--display none`. All combinations use the same capture/inference loop; DRM requires the Picamera2 camera backend.

Display adapters live in presentation. DRM uses optional preview/overlay methods on the Pi adapter, without adding Picamera2 details to the domain camera protocol or application orchestration. Its camera image is displayed natively by Picamera2, with landmarks and metrics drawn on a transparent RGBA overlay passed to `set_overlay()`. Camera pixels are not copied into that overlay.

Shared defaults in `config/settings.py` are **848×480 preview/capture** and **480×270 inference**, independently configurable with `--display-width`, `--display-height`, `--inference-width`, and `--inference-height`. OpenCV requests the preview dimensions from the webcam; the MediaPipe adapter resizes its input only when necessary, leaving the original preview frame intact. Actual webcam resolution depends on device support.

Picamera2 configures a main preview stream and a lores inference stream. `Camera.read()` returns lores as BGR; DRM displays main. On the Pi 3, lores uses YUV420, converted to BGR with stride padding excluded and no software resize. Both streams share the main crop, so normalized landmarks map onto the full display-size RGBA overlay despite their slightly different aspect ratios. These dimension defaults also apply in no-display mode.

`--orientation` accepts `normal` (default), `rotate180`, `mirror-horizontal`, or `mirror-vertical`. Picamera2 applies libcamera transforms to camera configuration, affecting both streams. OpenCV applies the corresponding flip to captured frames before inference and display. No 90°/270° rotation is implemented.

## Asynchronous preview

`LandmarkDetector` exposes `submit(frame)` and `snapshot()`. `LandmarkPreview.next_frame()` reads a camera frame, submits it, and immediately retrieves the latest completed result. It never waits for inference to finish; startup frames may have no landmarks, and several frames may reuse a result. Landmarks can therefore lag a moving face.

The MediaPipe adapter uses Tasks `RunningMode.LIVE_STREAM` and `detect_async()`. Its callback updates only one latest-result slot, a timestamp, and a completion count. Conversion to domain landmarks happens on the main loop once per new result. MediaPipe may drop submissions when busy; the project creates no frame queue or custom inference worker. Optional blendshapes and facial transformation matrices are disabled, with one face configured by default.

Capture and display run on the main loop. The OpenCV backend requests 30 FPS; achieved capture/display rates depend on the device and processing overhead. About 30 FPS capture/display and at least 15 completed inferences per second on the Pi are goals, not verified Pi performance.

The overlay reports:

- **Capture FPS:** successful camera reads per second.
- **Display FPS (OpenCV only):** completed preview iterations (`imshow` plus the GUI event pump) per second; this does not measure monitor refresh.
- **Overlay updates/s (DRM only):** successful overlay submissions per second; this does not measure native camera preview FPS or screen refresh.
- **Inference FPS:** completed callback results per second, including results with no face.
- **Submitted / completed:** cumulative successful `detect_async()` calls and callback completions. Submitted frames may be dropped internally; their difference also includes work still in flight.
- **Result latency:** approximate submission-to-callback milliseconds for the latest result, including input conversion and dispatch. It excludes camera acquisition, rendering, and the age of a reused result.

Rates update over measured intervals of at least one second and initially show zero. No-display mode prints capture/inference rates, submitted/completed counts, and result latency periodically to the console. It performs no frame or overlay rendering and reports no display rate. Settings centralize the requested camera FPS, metrics interval, resolution, confidences, model paths, and renderer defaults.

Shutdown clears DRM overlays with `set_overlay(None)` and stops the native preview before closing the detector and camera. OpenCV windows are destroyed only in OpenCV display mode. Ctrl+C and pipeline errors follow the same cleanup path.

## Windows laptop setup

From the project root in PowerShell, with Python 3.11 on `PATH`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python tools/download_face_landmarker.py
python -m face_recognition.presentation.cli.recognize --camera opencv --display opencv
```

For OpenCV and DRM displays, benchmark rendering with `--landmarks none`, `--landmarks all` (the default), or `--landmarks selected --indices INDEX,INDEX,...`. Selected mode requires your explicit nonnegative landmark indices; out-of-range indices are skipped. No recognition subset is predefined. All landmark rendering modes retain metrics. Rendering fewer points measures rendering/post-processing overhead; it does **not** reduce MediaPipe neural-network inference cost. With `--display none`, all drawing is skipped regardless of landmark rendering mode.

The model download uses the [official MediaPipe Face Landmarker bundle](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/index#models). It is stored at `assets/face_landmarker.task` and is not committed. Set camera index, resolution, and detector confidence in `src/face_recognition/config/settings.py`. Run hardware-free checks with `$env:PYTHONPATH='src'; python -m unittest discover -s tests -p 'test_*.py'`.

If `python` resolves to the Windows Store alias, call your Python 3.11 interpreter by its full path. With the existing Windows Conda environment, activate `face-recognition`, set `$env:PYTHONPATH='src'`, and run the module command above. Install the pinned runtime dependencies with `python -m pip install -r requirements.txt` if needed.

## Raspberry Pi environment

Ensure Picamera2 0.3.31, libcamera, and python3-libcamera are available through Debian/Raspberry Pi OS apt/system packages. These are prerequisites, not pip requirements. With the system Python 3.11 interpreter, create the environment from the project root:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements-rpi.txt
python tools/download_face_landmarker.py
```

`--system-site-packages` is required so the environment can access the system-provided Picamera2 and libcamera Python bindings. The `Picamera2Camera` adapter imports Picamera2 and libcamera only when instantiated on Linux. Pi stream dimensions must be positive even integers, and lores dimensions must not exceed main dimensions.

From the project root, with `.venv` activated, use the upside-down camera and attached 848×480 screen through Picamera2 DRM (480×270 inference):

```bash
PYTHONPATH=src python -m face_recognition.presentation.cli.recognize --camera picamera2 --display drm --orientation rotate180
```

This explicitly selects `Preview.DRM` at screen origin (0, 0), with the main stream and overlay covering 848×480. Matching the preview stream to the attached screen avoids the former 4:3 pillar-boxing. It requires an attached display and access to KMS/DRM, without X11, Wayland, Qt, or a `DISPLAY` variable. The adapter replaces Picamera2's initial null-preview event loop with DRM. See the [Picamera2 preview and overlay documentation](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf).

To benchmark capture/inference without display rendering:

```bash
PYTHONPATH=src python -m face_recognition.presentation.cli.recognize --camera picamera2 --display none --orientation rotate180
```

Both Pi modes exit with Ctrl+C. No-display mode keeps Picamera2's non-visible event loop for camera capture. Hardware-free tests cover display selection, overlays, counters, and cleanup; actual DRM execution and performance on Pi hardware still require validation.

Create a separate environment on each machine: virtual environments contain platform-specific interpreters and binary packages and cannot be copied between Windows and Linux. Share source, configuration, compatible model assets, and the pinned requirements.

## Future work

Dataset capture; experimentally select a smaller landmark subset; normalize translation, scale, and possibly rotation; derive geometric features; compare KNN, SVM, and Random Forest; reject unknown people; validate the Pi camera on hardware; benchmark accuracy and throughput. These are not implemented yet.
