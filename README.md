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

The application depends on `Camera` and `LandmarkDetector`, not a camera library. The CLI selects `--camera opencv` (default) or `--camera picamera2`, plus `--display opencv` (default), `--display drm`, or `--display none`. All combinations use the same LIVE_STREAM capture/inference loop; DRM requires the Picamera2 camera backend.

Display adapters live in presentation. DRM uses optional preview/overlay methods on the Pi adapter, without adding Picamera2 details to the domain camera protocol or application orchestration. Its camera image is displayed natively by Picamera2, with landmarks and metrics drawn on a transparent RGBA overlay passed to `set_overlay()`. Camera pixels are not copied into that overlay.

Shared defaults in `config/settings.py` are **848×480 preview/capture** and **480×270 inference**, independently configurable with `--display-width`, `--display-height`, `--inference-width`, and `--inference-height`. OpenCV requests the preview dimensions from the webcam; the MediaPipe adapter resizes its input only when necessary, leaving the original preview frame intact. Actual webcam resolution depends on device support.

Camera adapters return a vendor-independent `Frame` containing opaque pixel data, logical dimensions, and a `PixelFormat`. OpenCV supplies BGR pixels. Picamera2 supplies the untouched lores YUV420/I420 buffer, including stride padding; `read()` does no color conversion. The domain and application contain no camera-library types or pixel conversion logic.

Picamera2 configures a main preview stream and a lores inference stream; DRM displays main independently. Only after LIVE_STREAM reserves its inference slot does the detector convert lores directly from YUV420 to RGB. Shared preprocessing converts the full packed planes before cropping RGB stride padding, with no resize at the configured 480×270 inference size. Accepted OpenCV frames retain BGR→RGB conversion and inference resizing when needed. Busy frames perform neither conversion nor resizing. Both Pi streams share the main crop, so normalized landmarks map onto the full display-size RGBA overlay despite their slightly different aspect ratios. These dimension defaults also apply in no-display mode.

DRM and no-display modes never convert captured camera pixels for presentation. If Picamera2 is explicitly paired with OpenCV display, the renderer separately converts YUV to BGR for that preview; Windows BGR preview needs no display conversion. Conversion-count and padded-plane tests run without Pi hardware. Throughput improvements still require measuring on the Pi.

`--orientation` accepts `normal` (default), `rotate180`, `mirror-horizontal`, or `mirror-vertical`. Picamera2 applies libcamera transforms to camera configuration, affecting both streams. OpenCV applies the corresponding flip to captured frames before inference and display. No 90°/270° rotation is implemented.

## Asynchronous preview

`LandmarkDetector` exposes `submit(frame)` and `snapshot()`. `LandmarkPreview.next_frame()` reads a camera frame, submits it, and immediately retrieves the latest completed result. It never waits for inference to finish; startup frames may have no landmarks, and several frames may reuse a result. Landmarks can therefore lag a moving face.

The production adapter uses Tasks `RunningMode.LIVE_STREAM` and `detect_async()`, with **at most one inference request in flight**. Submission atomically reserves a slot before preprocessing. While busy, new frames are counted as skipped and return immediately, without resizing, color conversion, image construction, or another MediaPipe call. A matching callback (including no face) releases the slot; the next camera iteration can submit its current frame. Submission errors also release it. There is no frame queue, latest-frame buffer, custom inference worker, or fixed inference-rate limiter.

The callback updates only the latest-result slot, timestamp, completion count, and in-flight state. Conversion to domain landmarks happens on the main loop once per new result. No state lock is held during MediaPipe calls. Shutdown disables callback updates and waits for any preprocessing/dispatch already underway before closing MediaPipe. Optional blendshapes and facial transformation matrices remain disabled, with one face configured by default. This bounds outstanding requests; actual inference speed and freshness on Pi still require benchmarking.

Capture and display run on the main loop. The OpenCV backend requests 30 FPS; achieved capture/display rates depend on the device and processing overhead. About 30 FPS capture/display and at least 15 completed inferences per second on the Pi are goals, not verified Pi performance.

The LIVE_STREAM overlay reports:

- **Capture FPS:** successful camera reads per second.
- **Display FPS (OpenCV only):** completed preview iterations (`imshow` plus the GUI event pump) per second; this does not measure monitor refresh.
- **Overlay updates/s (DRM only):** successful overlay submissions per second; this does not measure native camera preview FPS or screen refresh.
- **Inference FPS:** completed callback results per second, including results with no face.
- **Captured:** cumulative successful camera reads, including frames skipped by inference.
- **Submitted:** cumulative successful `detect_async()` calls. Busy skips and failed preprocessing/dispatch do not increment this count.
- **Completed:** cumulative matching result callbacks, including callbacks with zero faces; late callbacks after shutdown are ignored. At normal camera-loop snapshots, submitted minus completed is zero or one.
- **Skipped Busy:** cumulative frames rejected before preprocessing because inference or its submission call is still active. These frames are never retained for later processing.
- **Result latency (callback):** callback monotonic time minus that frame's monotonic submission timestamp, in milliseconds. The timestamp is taken before inference preprocessing, so this includes resizing (if needed), conversion and dispatch, but excludes capture and rendering. The value stays fixed while a result is reused.
- **Result age:** current monotonic time minus the latest completed result's original submission timestamp, in milliseconds. This includes callback latency plus time since completion and grows when the displayed result is reused. Both timing metrics show `pending` before the first completion.

Diagnostic `Timing ms` values describe the **latest completed inference**, not averages. Accepted frames record monotonic nanosecond timestamps at slot reservation, after preprocessing/image construction, immediately after `detect_async()` returns, and on callback entry. Busy frames record no diagnostic timestamps; no per-frame logging is added. Console and DRM/OpenCV metrics show:

- **prep / `preprocessing_ms`:** image-ready time minus slot-reservation time, including conversion, any resize, contiguous storage preparation, and MediaPipe image construction.
- **dispatch / `dispatch_call_ms`:** API-return time minus image-ready time, measuring the synchronous `detect_async()` call.
- **async / `async_result_ms`:** callback-entry time minus API-return time, including asynchronous processing and callback scheduling. This is not a measurement of neural-network execution alone.
- **total / `total_callback_latency_ms`:** callback-entry time minus slot-reservation time; equals prep + dispatch + async.

The breakdown stays fixed while the result is reused. Existing result-age semantics and legacy `Result latency` remain unchanged; legacy latency uses the integer millisecond timestamp supplied to MediaPipe, so it can differ slightly from the precise diagnostic total. If a callback arrives before `detect_async()` returns, async is negative under the stated definition; it is not clamped. Dispatch/async remain pending until the API-return timestamp is available. Timing arithmetic is performed outside the callback.

Rates update over measured intervals of at least one second and initially show zero. No-display mode prints capture/inference rates, captured/submitted/completed/skipped-busy counts, callback latency, and result age periodically to the console. The same counters and timing metrics appear on DRM/OpenCV overlays. No-display mode performs no frame or overlay rendering and reports no display rate. Settings centralize the requested camera FPS, metrics interval, resolution, confidences, model paths, and renderer defaults.

DRM sends one initial overlay, then schedules updates independently from camera capture. In `all` and `selected` modes, each new completed result timestamp triggers an update (including no-face results); repeated frames with the same result do not. Metrics also trigger a refresh at `metrics_interval_seconds` (default one second), preserving the latest landmarks. In `none` mode, only the initial overlay and periodic metrics refreshes are sent. Overlay allocation/drawing occurs only for these updates, and overlay counts/rates measure successful `set_overlay()` submissions, excluding cleanup. Native DRM camera video continues independently; the capture/inference loop does not sleep or wait for the overlay timer. OpenCV display behavior is unchanged.

Shutdown clears DRM overlays with `set_overlay(None)` and stops the native preview before closing the detector and camera. OpenCV windows are destroyed only in OpenCV display mode. Ctrl+C and pipeline errors follow the same cleanup path.

## Raspberry Pi architecture decision

VIDEO / `detect_for_video()` with a latest-frame worker was experimentally evaluated on Raspberry Pi 3B+. With a continuously visible active face, it sustained approximately 7–8 inference FPS and 125–135 ms processing latency. LIVE_STREAM sustained approximately 7–8 inference FPS with 114–120 ms total callback latency under the same workload. LIVE_STREAM with single-in-flight backpressure was retained for comparable/slightly better performance, simpler architecture, and lower result age. The experimental runtime has been removed; its implementation is preserved in Git history/tag. The earlier approximately 25 ms VIDEO measurements without an active face were not representative of the intended workload.

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
PYTHONPATH=src python -m face_recognition.presentation.cli.recognize --camera picamera2 --display drm --orientation rotate180 --landmarks all
```

This explicitly selects `Preview.DRM` at screen origin (0, 0), with the main stream and overlay covering 848×480. Matching the preview stream to the attached screen avoids the former 4:3 pillar-boxing. It requires an attached display and access to KMS/DRM, without X11, Wayland, Qt, or a `DISPLAY` variable. The adapter replaces Picamera2's initial null-preview event loop with DRM. See the [Picamera2 preview and overlay documentation](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf).

To benchmark DRM with metrics only (no landmark drawing):

```bash
PYTHONPATH=src python -m face_recognition.presentation.cli.recognize --camera picamera2 --display drm --orientation rotate180 --landmarks none
```

To benchmark capture/inference without display rendering:

```bash
PYTHONPATH=src python -m face_recognition.presentation.cli.recognize --camera picamera2 --display none --orientation rotate180 --landmarks none
```

Both Pi modes exit with Ctrl+C. No-display mode keeps Picamera2's non-visible event loop for camera capture. Hardware-free tests cover display selection, overlays, counters, and cleanup; actual DRM execution and performance on Pi hardware still require validation.

Create a separate environment on each machine: virtual environments contain platform-specific interpreters and binary packages and cannot be copied between Windows and Linux. Share source, configuration, compatible model assets, and the pinned requirements.

## PC normalization experiment

This separate experimental CLI reuses the OpenCV camera, LIVE_STREAM detector, `LandmarkPreview`, performance tracker, and raw landmark renderer. It does not change the production preview. From the project root, with the Python 3.11 environment activated and `assets/face_landmarker.task` available:

```powershell
$env:PYTHONPATH='src'
python -m face_recognition.presentation.cli.normalize_landmarks
```

It uses `Settings` for the camera and inference dimensions (default 848×480 capture, 480×270 inference). Two windows show the original camera/landmarks and a synthetic identity-normalized cloud. The cloud has a **fixed 640×640 canvas**, X/Y range **[-2, 2]**, and equal scale: `pixel = round((coordinate + 2) * 639 / 4)` on each axis, with positive Y down. Eye-center guides are at (-0.5, 0) and (+0.5, 0); points outside the range are clipped. No per-face fitting, recentering, or zoom changes normalized values. Z is used for comparison, not drawing.

Press **r** to capture/replace the latest completed face as an in-memory reference; **q** or Ctrl+C quits. No-face results prevent reference replacement. Current points are green and reference points gray. State/normalization/RMSE update only when a newer completed `timestamp_ms` arrives; reused snapshots are not new observations. Only the first detected face is compared, assuming the same subject and corresponding landmark indices. Invalid normalization or unequal landmark counts show an unavailable comparison rather than an identity decision.

Raw comparison corrects only image aspect ratio: `(X, Y, Z) = (x, y * image_height / image_width, z)`, without eye centering, scale, or roll correction. Both raw and normalized RMSE use `sqrt(sum((x-x_ref)^2 + (y-y_ref)^2 + (z-z_ref)^2) / N)` across equal ordered landmark counts. Raw errors use image-width units; normalized errors use interocular units, so inspect trends within each measure rather than treating their numerical ratio as a recognition score. Reference capture gives zero errors; all reference data disappears on exit and nothing is written to disk.

Manually compare neutral posture, left/right/up/down translation, closer/farther movement, and positive/negative roll. Normalization should reduce these geometric variations; it does not correct yaw, pitch, expressions, landmark jitter, or camera noise. There are no recognition thresholds or final identity features in this experiment.

### Guided statistical validation

Run the separate PC session from PowerShell with the same Python 3.11 environment and task model:

```powershell
$env:PYTHONPATH='src'
python -m face_recognition.presentation.cli.validate_normalization
```

Instructions and progress appear in the terminal; the existing raw camera preview remains visible. Keep the same subject throughout. Finish repositioning **before SPACE**, then hold the pose still. SPACE starts each stage separately (no Enter needed); **q** or Ctrl+C exits. SPACE/q also work in the camera window if it has focus. Terminal key polling uses the Windows console; on other PCs use the camera-window controls.

`ValidationConfig` groups the defaults: **5 warm-up completions**, then **30 valid measurement results** per stage. Only newer completed `timestamp_ms` values count, including across stage boundaries. Warm-up discards completions even with no face; measurement skips missing faces, invalid normalization, nonfinite coordinates, or unequal landmark counts without filling a slot. Collection may therefore take longer than 30 inference intervals. The first detected face is used, with corresponding landmark indices assumed.

REFERENCE collects 30 samples and builds separate raw/normalized templates by averaging each corresponding X/Y/Z coordinate independently. Raw coordinates use `(x, y * image_height / image_width, z)` only; normalized samples use the configured `LandmarkNormalizer`. **CENTER then collects an independent sample set**, rather than reusing the reference samples.

The 13 tests run in order: CENTER; TRANSLATION LEFT, RIGHT, UP, DOWN; NEAR; FAR; ROLL LEFT, RIGHT; YAW LEFT, RIGHT; PITCH UP, DOWN. Each measurement's raw/normalized RMSE against its respective reference template uses the definition above. Each test prints mean, **population standard deviation** (variance divided by sample count), median, minimum, and maximum. A final terminal table includes means, standard deviations, and medians. Raw and normalized columns use different units and are separate stability baselines; no pass/fail thresholds are applied.

Templates, temporary samples, and summaries exist only in memory. Nothing is saved. This experiment reuses the existing pipeline without changing production scheduling, preprocessing, cameras, display backends, performance tracking, or feature extraction.

## Future work

`LandmarkNormalizer(image_width, image_height).normalize(landmarks)` provides standalone, hardware-free identity normalization. It first corrects MediaPipe's image-normalized coordinates to width-relative units `(x, y * image_height / image_width, z)`, then applies eye-centered translation, roll correction and interocular scaling. Eye centers use corners 33/133 and 362/263; z is centered/scaled only. Tests use synthetic pixel-space geometry encoded for non-square images, including 480×270. It is used by the separate PC experiment, not the production preview or feature extractor; its eye-relative output is distinct from the input image-normalized representation.

Remaining work: dataset capture; experimentally select a smaller landmark subset; derive geometric features; compare KNN, SVM, and Random Forest; reject unknown people; validate the Pi camera on hardware; benchmark accuracy and throughput. These are not implemented yet.
