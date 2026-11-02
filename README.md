# Face Recognition

Layered Python project for facial landmark detection now and experimental identity recognition later. Development starts on a Windows x86_64 laptop; the intended later target is a Raspberry Pi 3B+ (Linux ARM64/aarch64) with a Raspberry Pi Camera Module 3.

**Current milestone:** OpenCV webcam or Picamera2 camera → asynchronous MediaPipe Tasks Face Landmarker → OpenCV window, native DRM preview, or console benchmark metrics. Press `q` in the OpenCV window or Ctrl+C to quit. Face detection and landmarks do not identify a person.

**Offline experiment status:** V1 and V2 have completed independent final evaluation and are experimentally closed. See [archived V1/V2 results](#archived-v1v2-experiments). Final results must not be used to retune either model version.

## Stack and architecture

**Python 3.11 is the only currently supported Python version** (`>=3.11,<3.12`). The dependencies are NumPy 1.26.4, opencv-contrib-python 4.10.0.84, MediaPipe 0.10.14, protobuf 4.25.9, and scikit-learn 1.7.2. The same versions are pinned in `pyproject.toml` and `requirements.txt`; `requirements-rpi.txt` includes those shared requirements.

The reference Raspberry Pi environment is a Raspberry Pi 3B+ running Debian 12 Bookworm on aarch64 with Python 3.11.2. It supplies Picamera2 0.3.31, libcamera, and python3-libcamera through system packages. Windows uses the shared Python dependencies without Picamera2. Scikit-learn is used by reusable offline classifier/feature-transform primitives; the production preview does not import it. The new offline dependency has not been validated on Pi hardware.

```text
src/face_recognition/
  domain/          Models and camera, detector, extractor, classifier contracts
  application/     Frame orchestration, dataset capture/processing, validation, development folds/evaluation
  infrastructure/  OpenCV and optional Picamera2 cameras, MediaPipe detector,
                   normalization/XYZ features, JSONL datasets, reusable sklearn pipelines and scorers
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

## Raw landmark dataset capture

Raw MediaPipe image-normalized landmarks are the **canonical stored samples**. Normalization, the all-coordinate baseline, future subsets, and geometric features can be derived later from this same dataset. Capture stores no images/video, normalized landmarks, feature vectors, or predictions. Use pseudonymous person IDs such as `p001`; names are unnecessary. IDs allow 1–64 ASCII letters/numbers/underscores/hyphens and reject traversal and Windows reserved device names.

From the project root, with the Python 3.11 environment activated and the existing task model available, capture a Windows/OpenCV trial:

```powershell
$env:PYTHONPATH='src'
python -m face_recognition.presentation.cli.capture --data-root "..\face-recognition-data\data\raw" --person-id p001 --session-id pc_trial_01 --camera opencv --display opencv
```

For the upside-down Raspberry Pi camera and attached DRM screen:

```bash
PYTHONPATH=src python -m face_recognition.presentation.cli.capture --data-root ../face-recognition-data/data/raw --person-id p001 --session-id session_01 --camera picamera2 --display drm --orientation rotate180
```

Use `--display none` for capture without a visual preview (still run in an interactive terminal). Existing OpenCV/Picamera2 cameras and displays are reused; Picamera2 remains optional on Windows. Source metadata records the backend/device, orientation, and stream sizes. Camera index, inference/display dimensions, and landmark drawing modes are configurable; rendering selections do not reduce the landmarks stored.

Each condition waits for **SPACE in the terminal** (no Enter). **q** or Ctrl+C cancels; q also works in the OpenCV window. Move into the requested pose first, then maintain it with small natural variations rather than large continuous movements. Defaults are **5 new warm-up completions**, **20 accepted samples**, and **at least 250 ms between accepted detector timestamps within each condition**. Repeated snapshots do not count; no face, multiple returned faces, inconsistent counts, and nonfinite coordinates do not fill a sample slot. Warm-up discards completions even without a face. The detector is configured for one face; the operator must keep only the intended participant in view, as there is no identity check.

Override these defaults with `--warmup-results`, `--samples-per-condition`, and `--minimum-sample-interval-ms`. The 13 conditions are CENTER; TRANSLATION LEFT, RIGHT, UP, DOWN; NEAR; FAR; ROLL LEFT, RIGHT; YAW LEFT, RIGHT; PITCH UP, DOWN. Defaults yield **260 samples/session**; no fixed number of people or sessions is enforced.

```text
data/raw/<person_id>/<session_id>/
  manifest.json
  center.jsonl
  translation_left.jsonl
  translation_right.jsonl
  translation_up.jsonl
  translation_down.jsonl
  near.jsonl
  far.jsonl
  roll_left.jsonl
  roll_right.jsonl
  yaw_left.jsonl
  yaw_right.jsonl
  pitch_up.jsonl
  pitch_down.jsonl
```

Each JSONL line uses schema version 1 (shortened illustrative landmark list):

```json
{"schema_version":1,"person_id":"p001","session_id":"session_01","condition":"center","sample_index":0,"detector_timestamp_ms":123456789,"inference_width":480,"inference_height":270,"landmarks":[[0.1,0.2,-0.03],[0.2,0.3,-0.02]]}
```

Coordinates are preserved without rounding or aspect-ratio correction, in landmark-index order with X/Y/Z per point. The landmark count comes from the detector. Samples retain the actual configured inference dimensions needed for later normalization. Detector timestamps are monotonic submission times, not wall-clock capture dates; they are not assumed comparable across process restarts.

The manifest records `schema_version`, `person_id`, `session_id`, UTC `created_at`, `camera_backend`, `camera_device_id`, `orientation`, `requested_camera_fps` (null for Picamera2), inference/display dimensions, `detector_model_sha256`, `mediapipe_version`, detection/tracking confidences, `landmark_coordinate_space`, `detector_timestamp_clock`, `expected_landmark_count`, ordered `conditions`, sampling configuration, and `completed_conditions`. The coordinate description explicitly states **raw image-normalized**, not identity-normalized.

Only a full condition batch is published. JSONL data is written/fsynced to `.jsonl.tmp`, then atomically published without replacing an existing condition using a same-filesystem hard link (NTFS/ext4), and the temporary name is removed. The manifest is then updated through a flushed temporary file and atomic replacement. Cancelled in-memory batches are discarded; partial temporary files never count as completed. Linux directory updates are also fsynced.

Run the **same command/configuration** to resume: metadata, model digest/version, protocol, and completed files are validated, completed conditions are skipped, and an unfinished condition restarts with SPACE and warm-up. A complete unmarked file left between publication and manifest update is validated and recovered; incomplete/corrupt final files or incompatible configuration fail clearly. Use a new session ID to change capture configuration. Run one capture process per session.

Collect multiple independent sessions per person (approximately three is a later recommended protocol). **Train/test separation must be by session**, never by randomly splitting adjacent frames from one capture burst. Capture performs no splitting or model fitting. Use an external `--data-root` for biometric capture. Repository-local generated contents under `data/raw/` remain ignored by Git as a safeguard; `.gitkeep` preserves the directory.

## Offline processing of external datasets

Keep biometric datasets **outside this Git repository**, including all derived normalized landmarks and feature vectors. Expected sibling layout:

```text
Machine Learning/
├── face-recognition/        source code and tests
└── face-recognition-data/
    └── data/
        ├── raw/            canonical raw landmarks; never modified by processing
        └── processed/      generated derived representations
```

From the repository root in PowerShell, with the Python 3.11 environment activated:

```powershell
$env:PYTHONPATH="src"
python -m face_recognition.presentation.cli.process_dataset `
  --raw-root "..\face-recognition-data\data\raw" `
  --output-root "..\face-recognition-data\data\processed"
```

Both arguments are required and accept relative or absolute paths. Relative paths resolve from the current working directory. The raw root must exist. **The output root must not already exist, even if empty**; it must be outside the repository and must not overlap the raw tree in either direction. Use a new output directory for another run; nothing is overwritten or appended.

The command reuses the read-only reader, existing normalizer, baseline extractor, and transactional processed repository. It uses each sample's own inference dimensions and preserves participant/session/condition/sample provenance. It processes only manifest-declared completed conditions. Separate `normalized_landmarks_v1/` and `all_xyz_v1/` directories retain `<person>/<session>/<condition>.jsonl`, with processing manifests and an aggregate run manifest. Normalization and feature generation happen only in the existing application/components, not in the CLI.

Successful output contains aggregate participant/session/file/sample counts, representation names, and the resolved output directory. Errors are sanitized and sent to stderr; coordinates, feature-vector values, and raw records are never printed. Exit codes: **0** success, **1** input/processing failure or interruption, **2** argument/usage error. Failures stop the run; unpublished staging directories may remain with failed metadata. There is no automatic resume or deletion of failed staging data.

Raw sources remain unchanged. Normalized landmarks and flattened features are still **biometric-derived artifacts and must not be committed to Git**. Repository-local raw/processed/model directories remain ignored, and the processing repository also refuses repository-local output paths.

## Offline normalization quality evaluation

Evaluate the existing raw and `normalized_landmarks_v1` representations without changing either input. `all_xyz_v1` is not required. From the repository root in PowerShell:

```powershell
python -m face_recognition.presentation.cli.evaluate_normalization `
  --raw-root "..\face-recognition-data\data\raw" `
  --processed-root "..\face-recognition-data\data\processed"
```

For each person/session separately, the evaluator builds a mean corrected-raw center template and a mean stored-normalized center template from all available center samples. Each condition's samples are compared only with that session's templates using the existing 3D per-landmark RMSE. Stored normalized data is **not normalized again**. Full provenance, dimensions, indices, counts and file catalogs must match; invalid input fails explicitly. Center statistics describe deviations from the same samples' mean (an in-sample baseline, not an independent centered test).

Raw geometry uses `(x, y * image_height / image_width, z)` only. Native raw RMSE has image-width units, while normalized RMSE has interocular units, so subtracting their native magnitudes would be misleading. The evaluator additionally computes **comparable raw RMSE = native raw RMSE / center-template interocular distance**, using the production normalizer's eye references. This is a fixed reference-unit conversion per session: non-center raw samples receive no eye centering, roll correction, or per-sample scale normalization.

Per-condition output includes native raw mean/std, comparable raw mean, normalized mean/std, sample pairs, signed absolute reduction, and percentage reduction. Reduction is `mean(comparable raw RMSE) - mean(normalized RMSE)`; percentage is `100 * reduction / mean(comparable raw RMSE)`, or `n/a` when its denominator is zero or the result is nonfinite. JSON also retains medians, extrema and comparable-raw std. Standard deviations are population statistics. Aggregates pool sample errors, so sessions with more samples contribute proportionally; face coordinates are never averaged across sessions or identities.

The targeted aggregate contains near/far, translations, and roll. Yaw/pitch are reported separately and have no guaranteed invariance. Negative reductions are preserved. There are **no pass/fail or recognition thresholds**.

Add `--report "..\face-recognition-data\reports\normalization_quality_v1.json"` to write an aggregate-only JSON report. The report has no coordinates, feature vectors, per-person measurements, identity mappings, or absolute input paths. Its target must be new and outside both input trees; publication is atomic without overwrite. Relative and absolute input/report paths are accepted. Exit codes are 0 success, 1 evaluation/report failure, and 2 usage error. Nothing is written unless `--report` is supplied.

## Session-based identity split

Create a **logical split of `all_xyz_v1` references**, not copied feature files. Randomly splitting adjacent samples would let the same session's lighting, pose, and capture conditions appear in both training and testing. Whole sessions therefore stay in one partition, keyed by `(person_id, session_id)`:

- `known_train_pool`: explicitly selected sessions of known identities. This is the only pool allowed for fitting feature selection, correlation filtering, PCA, StandardScaler, or classifiers. Future session-aware validation/model selection must stay inside this pool.
- `known_test`: explicitly held-out sessions of known identities. Never use them to fit preprocessing, select features, tune hyperparameters, or choose a classifier; reserve them for final known-identity evaluation.
- `unknown_pool`: **all sessions** of explicitly unknown identities. Keep it untouched by all training/model-selection operations. It is reserved for a later rejection stage and is **not yet a final test set**; this command does not subdivide it.

Roles are never inferred from session counts. Repeat `--known-id` / `--unknown-id` for every dataset identity, and repeat `--train-session` / `--known-test-session` for the global known-session policy. Every identity must be assigned exactly one role. Every known identity must have exactly the requested train/test sessions: missing or extra known sessions fail rather than silently disappearing. Unknown identities may have any number/names of sessions. There are no fixed identity/session/sample counts and no intentionally excluded identities in this version.

For an illustrative two-session training policy, this PowerShell command prompts for your actual pseudonymous ID lists without assuming their membership:

```powershell
$knownIds = @((Read-Host "Known IDs, comma-separated").Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$unknownIds = @((Read-Host "Unknown IDs, comma-separated").Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$splitArgs = @(
  "--processed-root", "..\face-recognition-data\data\processed",
  "--output", "..\face-recognition-data\splits\identity_split.json",
  "--train-session", "session_01",
  "--train-session", "session_02",
  "--known-test-session", "session_03"
)
foreach ($identityId in $knownIds) { $splitArgs += @("--known-id", $identityId) }
foreach ($identityId in $unknownIds) { $splitArgs += @("--unknown-id", $identityId) }
python -m face_recognition.presentation.cli.create_identity_split @splitArgs
```

The metadata reader shares processed manifest/path/catalog validation with the normalized reader. It scans inline feature scalars only to validate JSON, finiteness, and dimensions, discarding each value immediately; it never constructs feature vectors or invokes any ML fitting. Source files remain unchanged. Feature dimensions must be consistent across the representation, and inference dimensions consistent within each session. Final sample counts must match the completed representation manifest.

The versioned split JSON records explicit roles, session assignments, partition definitions, counts, leakage constraints, and ordered sample references (`person_id`, `session_id`, `condition`, `sample_index`, `partition`, and a path relative to the processed root). Ordering is lexical person/session order, then capture-protocol condition order and sample index. It contains no features, coordinates, real-name mapping, or personal absolute paths. Keep this artifact outside Git; the writer refuses repository-local and processed-tree destinations. Publication is atomic without overwriting an existing file. Successful CLI output is aggregate only; exit codes are 0 success, 1 validation/write failure, and 2 usage error.

The split command performs no training, fold generation, feature selection, learned preprocessing, rejection thresholds, or model evaluation. Later consumers must honor the recorded partitions; these logical rules are not filesystem access controls.

## Session-aware development folds

Only `known_train_pool` supplies development features. The default two-session example is:

```text
known_train_pool
├── session_01 → Fold A fit,        Fold B validation
└── session_02 → Fold A validation, Fold B fit
```

Session names come from the split, not hardcoded training logic. No random sample/frame split is performed. Every known identity must have all configured training sessions; each fold covers the whole development pool with disjoint fit/validation sessions and sample keys. **Feature selection, PCA, scaling and classifiers must be fitted only on the fit side of each fold.** Validation may only be transformed with parameters learned from fit. `known_test` remains the final holdout; `unknown_pool` remains reserved for later rejection work. Neither supplies development features.

The read-only loader validates the stored split and processed file catalog, then opens **only training-pool JSONL files**. Protected partitions are checked as metadata only; their payloads are not opened or revalidated. Training sample keys and counts must match the split; inference dimensions and finite feature values are also validated. No files are written.

`DevelopmentDatasetLoader(processed_root).load(read_identity_split(split_path))` returns a single read-only `float32` feature matrix, `int32` labels, a sorted `label_to_person` mapping and ordered sample metadata. Source precision on disk is unchanged. Rows are session-first; `create_session_folds(data)` returns immutable Fold A/B objects whose `fit.X`, `fit.y`, `validation.X` and `validation.y` share that matrix through read-only slices. Reprs omit biometric values. For other training-session counts, supply explicit `SessionFoldSpec` objects through the application API; noncontiguous row selections gather arrays on demand. Read-only array flags guard normal API use, not deliberate memory modification.

Inspect aggregate counts without fitting any model:

```powershell
python -m face_recognition.presentation.cli.inspect_training_folds `
  --processed-root "..\face-recognition-data\data\processed" `
  --split-manifest "..\face-recognition-data\splits\identity_split.json"
```

This diagnostics command requires exactly two training sessions for automatic opposite folds. It prints fit/validation identity and sample counts only; exit codes are 0 success, 1 input/loading/fold failure, and 2 usage error. This command performs no feature selection, learned preprocessing, classifier fitting or evaluation.

## Archived V1/V2 experiments

V1 and V2 are experimentally closed. Their executable experiment workflows,
model/config repositories, CLIs and experiment-only tests have been removed from
V3 HEAD. The complete history remains in Git; **`v2-open-set-final`** preserves
the completed V2 implementation and its historical operating instructions.

The following independent final holdout results are historical records only:

| Final metric | V1 | V2 |
| --- | ---: | ---: |
| Closed-set frame accuracy | 0.927692 | 0.878462 |
| Temporal ROC AUC | 0.736982 | 0.727761 |
| UNKNOWN false-acceptance rate (FAR) | 0.198718 | 0.416667 |
| KNOWN false-rejection rate (FRR) | 0.450000 | 0.280769 |
| Open-set exact accuracy | 0.644231 | 0.649038 |

V2 evaluated 10 known identities / 10 sessions / 2600 frames and 3 new unknown
identities / 6 sessions / 1560 frames. Its final closed-set macro F1 was 0.876944.
Its selected development temporal AUC/open-set accuracy were 0.846953/0.774231,
which did not transfer to the independent final population. Lower known false
rejection came with unacceptably high false acceptance on new unknown identities.
Calibration and development results are not final generalization measurements.

Neither archived model may be retuned from these final results. Changes to
normalization, features, rejection/threshold policy, classifier or temporal
policy belong to a new model version and experimental protocol.

## V3 development foundation

Current HEAD retains capture/preview and Raspberry Pi runtime support, raw and
processed datasets, normalization and its offline quality checks, all_xyz_v1
feature extraction, logical session splits and fit-only development loading.
Reusable sklearn classifier/feature-transform pipelines, centroid and neighbor
scorers, temporal windows/voting, rejection metrics, ROC/EER analysis, artifact
hashing and aggregate JSON publication remain available as primitives.

These primitives do not select a V3 model or train a recognizer. The nested
protocol below defines V3 data roles. The novelty comparison implementation is
available and the initial comparison has completed. The same-model calibration
refinement below has not yet run on real data. See [the cleanup inventory](docs/v3_archive_cleanup.md)
for the removed workflows and retained dependency boundaries.

`LandmarkNormalizer` and `LandmarkFeatureExtractor` keep their existing behavior.
Live rolling-window inference, deployment benchmarking and attendance logic are
not implemented by this cleanup.

## V3 nested cross-validation protocol

**V2 remains the last version with an independent final holdout.** No more data
can be collected. V3 therefore uses the complete available population in nested
cross-validation, rather than claiming a new independent final test. The generic
session-split examples above are separate tools, not the V3 nested protocol.
No archived V1/V2 workflow is restored.

An external experiment specification assigns all identities and expected session
names explicitly. The current experiment supplies 10 KNOWN identities with four
sessions each and 8 UNKNOWN identities with two sessions each. Every cataloged
identity must have exactly one role; no extra participant/session/condition is
silently excluded. Session/identity names are not algorithm constants. A complete
**synthetic** specification has this structure (supply all 10 actual KNOWN IDs
and all 8 UNKNOWN IDs for the real experiment):

```json
{
  "schema_version": 1,
  "experiment_id": "open_set_v3",
  "representation": "all_xyz_v1",
  "known_ids": ["known_a", "known_b"],
  "unknown_ids": ["novel_a", "novel_b", "novel_c", "novel_d", "novel_e", "novel_f", "novel_g", "novel_h"],
  "known_sessions": ["session_01", "session_02", "session_03", "session_04"],
  "unknown_sessions": ["session_01", "session_02"]
}
```

Four outer folds hold out each configured KNOWN session once for **every KNOWN
identity**. Lexically sorted UNKNOWN IDs are paired consecutively into four
disjoint groups; each complete pair, including both sessions of each identity,
is outer test exactly once. Outer fold ordering pairs the sorted known test
sessions with these sorted UNKNOWN groups. Each outer development population
thus has three KNOWN sessions and six UNKNOWN identities. Outer tests estimate
generalization and must never influence that fold's feature/normalization,
classifier, novelty strategy, hyperparameter or threshold decisions.

Inside each outer fold, use all **six permutations** of the three KNOWN sessions
as **fit / threshold calibration / validation**, one session per role. Cross
these with the **three remaining UNKNOWN-pair validation rotations**, giving
**18 inner splits per outer fold**. Four UNKNOWN identities calibrate and two
validate; UNKNOWN samples never fit preprocessing, identity classifiers or
novelty reference structures. ANOVA/scalers/classifiers/learned novelty steps may
fit only KNOWN inner-fit data. Thresholds may use only the disjoint KNOWN and
UNKNOWN inner-calibration roles. Inner validation compares strategies and must
not adjust their thresholds. The KNOWN validation session is absent from both
fit and calibration; UNKNOWN validation identities are absent from calibration.
Role IDs inherit all of their cataloged sessions/conditions as appropriate.

Repeated inner observations are not additional independent samples. Future
evaluation code must report outer results separately, retain each fold's
authorization boundary and construct any outer refit/threshold using only that
fold's development data. This commit implements **partitions only**: it does
not choose a model, fit anything, search thresholds, set a temporal window or
define the later outer-refit threshold-transfer procedure. Outer results cannot
be used to revise the selection process within that fold. Nested CV is internal
evaluation of the prespecified procedure, not a newly collected independent test.

Save the caller-provided specification externally, for example as
`../face-recognition-data/v3/splits/experiment_v3.json`, then create the protocol:

```powershell
python -m face_recognition.presentation.cli.create_v3_cv_protocol `
  --processed-root "..\face-recognition-data\v3\processed" `
  --experiment-spec "..\face-recognition-data\v3\splits\experiment_v3.json" `
  --output "..\face-recognition-data\v3\splits\cv_protocol_v1.json"
```

Add **`--validate-only`** to check an existing output against the current
specification/catalog without writing. Creation uses external atomic
no-overwrite publication, with no force option. Normal output is aggregate only.

Catalog inspection reuses completed `all_xyz_v1` processing metadata and safe
directory discovery. It checks error/rejection counts, exact roles/sessions,
all 13 expected capture conditions per session, nonempty condition files and
deterministic ordering. **No JSONL file is opened, hashed or decoded.** The
manifest status is `catalog_validated`, not a claim that individual records or
feature dimensions were inspected. A future payload loader must validate those
before model execution. Other processed representations are not protocol inputs.

The versioned manifest contains the normalized specification, observed sessions,
relative condition-file references/sizes, all four outer folds and their inner
splits, generation policy and SHA-256 digests. The processing-manifest digest
hashes its exact bytes; specification/catalog/protocol digests use canonical
metadata JSON. The protocol digest excludes only its own field. Reading a
protocol regenerates the expected folds, so rehashing an illegal role assignment
does not make it valid. Validate-only also detects changed source catalog
metadata. These are **metadata fingerprints, not biometric payload hashes**:
same-size feature-file edits require later record/content validation. No absolute
source paths, feature vectors, individual scores or predictions are persisted.

The protocol defines data roles only; execution is described below.

## V3 nested novelty-score comparison

This experiment isolates novelty detection from identity classification. Identity
is fixed to **all_xyz_v1 / 1434 inputs → SelectKBest(f_classif), k=448 →
StandardScaler → RBF SVC C=4, gamma=0.001, probability=False**. SVC uses OVR
decision scores, break_ties=False, cache_size=128 and random_state=42. The scaler
keeps its normal centering/scaling/copy behavior. There is no feature-count or
classifier search and no change to normalization or XYZ ordering.

Exactly four higher-is-KNOWN scores are compared:

| Candidate | Frame knownness score |
| --- | --- |
| `svm_top1` (control) | Maximum class-ordered SVM decision score |
| `svm_margin` | Largest minus second-largest SVM decision score |
| `centroid_distance` | Negative Euclidean distance to the **SVM-predicted** identity's fit-only centroid |
| `mahalanobis_distance` | Negative Mahalanobis distance to the **SVM-predicted** identity's fit-only distribution |

Mahalanobis references use per-class sklearn LedoitWolf covariance with
assume_centered=False and store_precision=False. Add a deterministic ridge
`1e-8 * max(1, trace(covariance)/dimension)` and use Cholesky solves, without an
unregularized sample-covariance inverse. The ridge covers completely constant
classes as well as high-dimensional/singular fits. No class centroid, covariance
or preprocessing step sees UNKNOWN or held-out known-session fitting data.
No combined score or additional novelty candidate is evaluated.

Every method uses **non-overlapping windows of 7**, inside person/session/condition,
ordered by sample index. Incomplete tails are counted and discarded. Confidence
is the median of all seven frame scores; identity uses the retained plurality
vote, all-frame decision-score sum for ties, then deterministic class order.
There is no consistency gate or window-size search. Closed-set KNOWN frame
metrics include trailing frames independently of rejection.

The existing protocol is authoritative. For each of its 18 inner splits, fit
preprocessing/SVM/references on the one KNOWN fit session; calibrate each method
on the separate KNOWN calibration session and four UNKNOWN calibration identities;
then validate at that unchanged threshold on the third KNOWN session and two
disjoint UNKNOWN validation identities. Threshold search enumerates unique scores,
finite interior midpoints and finite accept-all/reject-all boundaries. Acceptance
uses `score >= threshold`. Select higher exact integer balanced-accuracy numerator,
then lower FAR, lower FRR, then **higher numeric threshold**. AUC/EER remain
diagnostics and never replace an operational threshold.

Select one method per outer fold by **higher mean inner-validation balanced
accuracy**, lower mean FAR, lower mean FRR, higher mean AUC, then the table's
candidate order. Inner means weight the 18 splits equally. Inner comparisons are
development evidence, not independent final performance.

Before the outer refit, construct a threshold from **inner calibration windows
only**. For a physical `(person, session, condition, seven sample indices)` window,
deduplicate repeated observations from the same fit-session model, then take the
median score across distinct fit models. This happens **after** each model's
window-7 median. KNOWN windows contribute two distinct session-held-out models;
UNKNOWN windows contribute three. Repeated rotations produce six raw observations
per KNOWN window and twelve per UNKNOWN window, but each physical window enters
the final threshold search **once**. Reports distinguish raw, distinct-model and
unique physical-window counts. Pooling does not invent cross-model identity votes:
its calibration metrics are binary only. Do not average the 18 inner thresholds.

Refit the fixed identity pipeline and selected score references on all three
KNOWN outer-development sessions. Transfer the pooled threshold **directly and
unchanged**. Development-only diagnostics compare cross-fitted and final-model
KNOWN/UNKNOWN window distributions, mean/median shifts and acceptance-rate shifts.
A final development median outside cross-fitted p10/p90 raises a descriptive
warning; it does not change the method, normalize scores, recalibrate or reject a
result. KNOWN final-model development scores are explicitly **training
resubstitution diagnostics**, not generalization or threshold-calibration data.
Direct score-scale transfer is an assumption tested by the outer results, not a
guarantee. No outer-test value performs alignment or threshold selection.

Only after selection and threshold construction is the outer test scored once.
It reports closed-set identity accuracy/macro precision/recall/F1, temporal
acceptance/rejection, FAR/FRR, correct/incorrect accepted identity rates, exact
open-set accuracy, binary metrics and diagnostic AUC/EER. Report unweighted outer
fold means/population standard deviations, including **fold-level** AUC/EER;
do not pool raw outer scores from different models or score families. **The four
outer folds are V3's nested-CV generalization estimate. V3 has no independent
untouched final holdout; V2 remains the last version with one.**

```powershell
python -m face_recognition.presentation.cli.evaluate_v3_novelty `
  --processed-root "..\face-recognition-data\v3\processed" `
  --protocol "..\face-recognition-data\v3\splits\cv_protocol_v1.json" `
  --report "..\face-recognition-data\v3\reports\novelty_comparison_v1.json"
```

The command refuses an existing report before payload/model work. It validates
the protocol digests/regenerated roles and exact current catalog/processing
manifest before decoding payloads into one read-only float32 matrix. All records
must have 1434 finite values, correct identity/session/condition provenance,
contiguous ordered indices and increasing timestamps; session dimensions and
aggregate declared sample counts must agree. Payload bytes are fingerprinted
while decoding. File timestamps/catalog and protocol bytes are checked again
before publication. Integrity validation includes outer-test files, but their
values are not supplied to inner fitting, calibration or selection.

Execution is sequential with one native thread. Identical inner fit populations
reuse cached inference keyed by protocol/content digests, outer fold, exact fit
sessions and full fixed configuration. Three inner models are fitted per outer
fold, then released after scoring; thresholds are **never** cached across
calibration populations. One final outer model is fitted afterward. No trained
model or reference parameters are persisted. The external JSON is atomic,
no-overwrite and aggregate-only: provenance, configuration, inner summaries and
results, outer metrics, pooling/transport diagnostics and selection frequencies.
It contains no feature vectors, per-sample scores/predictions, centroids,
covariances, support vectors or absolute source paths. Elapsed times are omitted
so identical inputs produce deterministic report JSON in the same environment.

The initial comparison completed and its external `novelty_comparison_v1.json`
must remain immutable. It selected Mahalanobis in all four outer folds, and all
four reported threshold-transport shift warnings. These observations motivate the
structural calibration refinement below, not another novelty-method search.

## V3 same-model Mahalanobis calibration refinement

This is **V3 threshold-calibration refinement under cross-validation**. The
initial outer-CV results have already been inspected and motivated the revision;
it is **not a pristine untouched final evaluation or independent final
generalization estimate**. V3 has no additional untouched holdout. V2 remains
the last independent external holdout experiment.

Mahalanobis is now fixed. The all_xyz_v1/1434 → ANOVA-448 → StandardScaler →
RBF SVC C=4, gamma=0.001, probability=False pipeline, existing regularized
Mahalanobis implementation, non-overlapping window 7, median confidence,
plurality/tie rules and disabled consistency gate are unchanged. No alternative
novelty methods or inner strategy search run in this study.

The existing protocol continues to authorize identities, sessions, catalog and
outer tests. Derive a new calibration role cyclically: the next session after
the outer-test session in the globally ordered specification calibrates; the
other two development sessions fit. For the current session names:

| Outer fold | KNOWN fit sessions | KNOWN calibration | KNOWN test |
| --- | --- | --- | --- |
| outer_01 | session_03, session_04 | session_02 | session_01 |
| outer_02 | session_01, session_04 | session_03 | session_02 |
| outer_03 | session_01, session_02 | session_04 | session_03 |
| outer_04 | session_02, session_03 | session_01 | session_04 |

The table is derived, not hardcoded: each known session is test once, calibration
once and fit twice. **All six outer-development UNKNOWN identities**, with both
sessions, calibrate; the two outer-test UNKNOWN identities remain excluded.
Only the two KNOWN fit sessions fit the selector, scaler, SVM and class
distributions. UNKNOWN observations never fit a component.

Fit **one model per outer fold**, then freeze it before scoring calibration data.
Calibrate a single threshold on window-7 median Mahalanobis knownness scores from
the designated KNOWN session and UNKNOWN development identities. Reuse the exact
search: maximize binary balanced accuracy, then lower FAR, lower FRR, then higher
numeric threshold, over observed values, finite interior midpoints and finite
boundary cases. ROC AUC/EER are diagnostics and never replace this threshold.

**Do not refit anything after calibration.** The exact same selector, scaler,
SVM and covariance references score outer test at the calibrated threshold.
The sealed fold session checks fit-generation count, component identities and
an in-memory learned-state SHA-256 before/after scoring. It exposes no refit API,
permits one calibration and one test query per population role, and freezes the
threshold once. State fingerprinting streams serialization bytes into a digest;
no fitted parameters or model file are written. Reports explicitly record
`threshold_transport = "none_same_model_calibration"`, one fit per fold and
unchanged model-state digests.

```powershell
python -m face_recognition.presentation.cli.evaluate_v3_mahalanobis_calibration `
  --processed-root "..\face-recognition-data\v3\processed" `
  --protocol "..\face-recognition-data\v3\splits\cv_protocol_v1.json" `
  --selection-report "..\face-recognition-data\v3\reports\novelty_comparison_v1.json" `
  --report "..\face-recognition-data\v3\reports\mahalanobis_calibration_v1.json"
```

The required selection report supplies **method-selection provenance only**:
validate protocol/catalog, fixed configuration, four Mahalanobis selections and
source population metadata before payload/model work. Its payload digest must
match the strict V3 loader's current source bytes. Prior outer thresholds and
performance metrics are not calibration inputs. The original report and
protocol remain read-only and are fingerprinted again before publication.

The strict loader, input integrity checks and atomic external no-overwrite
publication are reused. Outer payloads are validated up front for integrity but
never enter fitting or calibration populations. A new aggregate-only report
records cyclic roles, calibration UNKNOWN IDs, calibration metrics and thresholds,
closed-set/open-set outer metrics, AUC/EER, frame/window/discard counts and
no-refit evidence. Four-fold means/population std and FAR/FRR min/max preserve
fold-level results; raw score distributions are never pooled across folds.
There are no feature vectors, per-sample scores/predictions, covariance matrices,
support vectors, real names or absolute source paths in the report. There is no
overwrite/force option or production-model persistence.

The same-model study has completed. Its external
`mahalanobis_calibration_v1.json` is immutable evidence, alongside the initial
novelty comparison and protocol.

## V3 post-hoc Mahalanobis diagnostics

The diagnostic workflow localizes errors in the completed same-model study;
it changes no model-selection or calibration decisions and makes no new final
performance claim. V3 remains a cross-validation-based calibration refinement.
V2 remains the last independent external holdout experiment.

The study retained aggregate metrics and learned-state digests, not fitted
models or individual observations. Diagnostics therefore **reconstruct each
original two-session fit in memory** using the unchanged pipeline. Before any
outer scoring, the reconstructed learned-state SHA-256 must equal the saved
digest. Apply the **saved threshold**, without calibration, threshold search,
ROC/EER threshold enumeration or strategy selection. Verify the model state
after inference and require the saved outer counts, closed-set/operational
metrics and score-distribution summaries to reproduce exactly. A mismatch in
data, software versions, learned state or saved summaries fails explicitly;
there is no fallback that changes the experiment. Grouped attributions are
reconstructed, not extracted from previously retained individual predictions.

Each fold reports KNOWN and UNKNOWN counts, acceptance/rejection rates and
mean/median window knownness by all 13 conditions and by pose, distance,
translation and center groups. KNOWN summaries also include correct accepted
identification and accepted misidentification. Additional tables cover KNOWN
test sessions, UNKNOWN identities and KNOWN condition-by-session effects.
Severity tables sort by FRR/FAR, then error count and label. Contribution tables
sort by error-window count and label, retaining all largest-count ties. Empty
groups have zero counts and null rates/statistics.

Pool decision counts only across disjoint outer tests. Do not pool raw scores
across fitted models: aggregate condition/group score summaries explicitly
report unweighted means/population standard deviations of fold means and
medians. Session/identity score summaries remain within their single test fold.

Sessions proxy time/place/environment context; conditions proxy
pose/distance/translation. **There are no lighting labels**, so these summaries
cannot identify lighting as a cause. Session, fold model and threshold effects
are confounded. The report's neutral interpretation describes observed
contributions and rates without causal or statistical-significance claims.

```powershell
python -m face_recognition.presentation.cli.analyze_v3_mahalanobis_diagnostics `
  --processed-root "..\face-recognition-data\v3\processed" `
  --protocol "..\face-recognition-data\v3\splits\cv_protocol_v1.json" `
  --evaluation-report "..\face-recognition-data\v3\reports\mahalanobis_calibration_v1.json" `
  --report "..\face-recognition-data\v3\reports\mahalanobis_diagnostics_v1.json"
```

The strict V3 loader and input integrity checks are reused. Source protocol,
evaluation and payload digests establish provenance; report references contain
filenames rather than absolute local paths. Inputs remain read-only. The new
external report is aggregate-only, atomic and no-overwrite, with no feature
vectors, individual scores/predictions or fitted parameters. No real diagnostic
conclusions are recorded here.

## V3 controlled pose-representation experiment

Inspected Mahalanobis diagnostics identified strong KNOWN rejection in yaw and
pitch conditions. This experiment varies **only the representation**:
`all_xyz_v1` remains the baseline; `all_xyz_pose_v2` is a deterministic face-local
coordinate candidate. Both retain all 478 XYZ landmarks (1434 values), in
landmark order. Yaw left/right, pitch up/**pitch down**, and roll left/right are
addressed by one geometric transform, without condition branches, identity
labels, fitted statistics or learned frontalization.

For each raw face, first correct aspect ratio as in the existing normalizer:
`p = (x, y * height/width, z)`. Use the same eye-corner groups
`E0=mean(33,133)` and `E1=mean(362,263)`. Let
`U=mean(10,109,338)` (forehead contour) and
`D=mean(152,148,377)` (chin contour). Averaging reduces reliance on one point;
these references still carry detector noise and may vary with expression.

```text
origin = (E0 + E1) / 2
scale = ||E1 - E0||_2                  # 3D interocular distance
x_hat = (E1 - E0) / scale
v = D - U
y_temp = v - dot(v, x_hat) * x_hat
y_hat = y_temp / ||y_temp||_2
z_hat = cross(x_hat, y_hat) / ||cross(x_hat, y_hat)||_2
y_hat = cross(z_hat, x_hat)            # reconstruct orthogonal vertical
q = (p - origin) / scale
pose_xyz = (dot(q, x_hat), dot(q, y_hat), dot(q, z_hat))
```

The frame is right-handed: +x runs from the first eye group to the second,
+y is superior-to-inferior, and +z is their cross product (approximately away
from the camera for a frontal image). There is no view-dependent sign flip.
Reject non-finite geometry/results, eye/vertical lengths <= `1e-8` in corrected
width-relative units, vertical residual <= `max(1e-8, 1e-6*||v||)`, or cross-product
norm <= `1e-6`. Check basis orthonormality and determinant +1 within `1e-12`.

This canonicalizes ideal rigid rotations of the supplied geometry.
**MediaPipe z is relative depth, not metric reconstruction.** This is
face-relative pose reduction, not true metric 3D frontalization or a guarantee
of pose invariance on real detections, occlusions or non-rigid expressions.

Process raw capture into a **fresh, separate pose-only root**. The command
preserves source provenance, timestamps, conditions and sample ordering/indices;
it publishes only `all_xyz_pose_v2` plus manifests, with the transform definition
recorded. Existing normalized/baseline datasets are never overwritten. Processing
fails on invalid geometry, retaining failed staging rather than a completed run.

```powershell
python -m face_recognition.presentation.cli.process_pose_dataset `
  --raw-root "..\face-recognition-data\data\raw" `
  --output-root "..\face-recognition-data\v3\pose_processed"

python -m face_recognition.presentation.cli.evaluate_v3_pose_representation `
  --baseline-root "..\face-recognition-data\v3\processed" `
  --candidate-root "..\face-recognition-data\v3\pose_processed" `
  --protocol "..\face-recognition-data\v3\splits\cv_protocol_v1.json" `
  --report "..\face-recognition-data\v3\reports\pose_representation_v1.json"
```

The existing protocol and cyclic two-fit/one-calibration/one-test session roles
are unchanged. Candidate catalog membership and every sample's provenance must
match the baseline; candidate manifests/bytes have separate fingerprints.
Strict width/finite-value/order/count checks remain in force. Both representations
use ANOVA-448, StandardScaler, RBF SVC C=4/gamma=0.001/probability=False and the
same regularized Mahalanobis implementation. Each representation/fold calibrates
its own numeric threshold on the same allowed populations using the existing
balanced-accuracy/FAR/FRR/higher-threshold rule. **No refit follows calibration**.
Windows remain non-overlapping size 7, median score, existing plurality/tie
rules, and no consistency gate. No classifier or novelty search occurs.

Report overall outer-fold means/population standard deviations, condition-level
KNOWN FRR/correct ID and score summaries, UNKNOWN FAR by condition/identity, and
KNOWN session FRR. All 13 conditions, including **pitch_down**, appear in fold
results and the CLI comparison table. Descriptive candidate-minus-baseline FRR
deltas explicitly cover yaw, pitch, roll, near and far. Condition decision counts
pool disjoint outer tests; raw score distributions are not pooled across models.

The predeclared ranking is higher mean outer binary balanced accuracy, lower
mean outer FRR, lower mean outer FAR, higher mean closed-set macro F1, then
`all_xyz_v1` on an exact tie. Improved yaw alone cannot select the candidate.
Reports are aggregate-only, atomic and no-overwrite; prior reports/protocols
remain unchanged. This is an **iterative V3 study motivated by inspected
diagnostics**, not an untouched final holdout or independent final generalization
claim. V2 remains the last independent external holdout.

## Frozen V3 open-set development configuration

The completed iterative comparison selected **all_xyz_pose_v2** under the
predeclared highest-mean-outer-balanced-accuracy rule. These are development
cross-validation results, **not independent final generalization**:

| Metric | all_xyz_v1 | all_xyz_pose_v2 |
| --- | ---: | ---: |
| Closed-set accuracy | 0.897500 | 0.948173 |
| Closed-set macro F1 | 0.897409 | 0.948020 |
| ROC AUC | 0.835956 | 0.890893 |
| EER | 0.229808 | 0.179567 |
| FAR | 0.120192 | 0.098558 |
| FRR | 0.302885 | 0.261538 |
| Open-set exact accuracy | 0.745879 | 0.782280 |
| Binary balanced accuracy | 0.788462 | 0.819952 |

Global improvement does not resolve pose sensitivity. KNOWN yaw-left FRR fell
from 0.725000 to 0.662500, and yaw-right from 0.862500 to 0.762500; both remain
difficult. Pitch-up did not improve (0.500000 to 0.512500); pitch-down remained
0.412500. Session/context variation and harder-to-reject UNKNOWN identities
(including p009 in prior diagnostics) remain limitations. MediaPipe relative
depth is not metric 3D geometry. These limitations motivate no tuning in the
freeze workflow.

Freeze the existing pose-transform definition and implementation fingerprint,
1434 inputs, ANOVA f_classif k=448, StandardScaler and RBF SVC
C=4/gamma=0.001/probability=False. Mahalanobis remains the only operational
novelty score, using the current per-class LedoitWolf/shrinkage, ridge and
Cholesky implementation and the SVM-predicted class; knownness is negative
distance. Preserve non-overlapping window 7, median score, existing
plurality/decision-sum/class-order identity ties, and disabled consistency gate.

**Same-model calibration is mandatory:** the fitted selector, scaler, SVM and
Mahalanobis references/covariances that produce calibration scores must also
use the resulting threshold. Record
`threshold_transport="none_same_model_calibration"`; no post-calibration refit
is allowed. Maximize binary balanced accuracy, then lower UNKNOWN FAR, lower
KNOWN FRR, then higher numeric threshold, using the existing enumeration.
AUC/EER remain diagnostic only. The freeze leaves the numeric threshold
uncalibrated and the deployable fit/calibration population allocation undefined;
the next stage must declare that allocation while preserving this principle.

```powershell
python -m face_recognition.presentation.cli.freeze_v3_open_set_config `
  --protocol "..\face-recognition-data\v3\splits\cv_protocol_v1.json" `
  --novelty-report "..\face-recognition-data\v3\reports\novelty_comparison_v1.json" `
  --calibration-report "..\face-recognition-data\v3\reports\mahalanobis_calibration_v1.json" `
  --diagnostics-report "..\face-recognition-data\v3\reports\mahalanobis_diagnostics_v1.json" `
  --pose-report "..\face-recognition-data\v3\reports\pose_representation_v1.json" `
  --output "..\face-recognition-data\v3\models\open_set_config_v3.json"
```

This metadata-only command validates the completed evidence chain and published
winner, hashes the exact bytes of all five artifacts, and fingerprints the local
normalization/Mahalanobis implementations. It never opens processed payloads,
fits an estimator, recalibrates, or reruns an experiment. The immutable,
deterministic configuration contains curated aggregate evidence, technical
filenames/digests, methodological status and limitations; no biometric vectors,
trained state or absolute paths. Publication is atomic and refuses overwrite,
with no force/unfreeze option. Changed decisions require a new version and
explicit development reasoning. **V3 has no additional untouched independent
holdout; V2 remains the last independently evaluated external holdout.**

## Deployable V3 construction and calibration

Construct the operational model from `open_set_config_v3.json`; do not repeat
selection, comparisons or cross-validation. Existing `all_xyz_pose_v2` payloads
are loaded directly, without regenerating normalization. Frozen protocol and
pose-report hashes authorize membership/catalog/payload provenance. By default,
the command locates their recorded filenames under the sibling `splits/` and
`reports/` directories; `--protocol`/`--pose-report` can locate relocated copies
only when the frozen hashes still match.

The fixed chronological allocation is:

| Role | Population | Sessions | Expected real frames |
| --- | --- | --- | ---: |
| Fit | All 10 registered KNOWN identities | session_01, session_02, session_03 | 7800 |
| KNOWN calibration | Same 10 registered identities | session_04 | 2600 |
| UNKNOWN calibration | All 8 development UNKNOWN identities | session_01, session_02 | 4160 |

Membership comes from the fingerprinted protocol; IDs are not guessed from
numbering. The real KNOWN set is p001/p004/p005/p006/p007/p010/p011/p012/p013/p014;
the UNKNOWN set is p002/p003/p008/p009/p015/p016/p017/p018. Uniform per-condition
counts are derived from the frozen catalog and checked against every actual
person/session/condition group (20 samples in the real catalog). Missing/extra
participants, sessions, conditions, rows or mismatched payload bytes fail closed.

Fit ANOVA-448, StandardScaler, RBF SVC C=4/gamma=0.001/probability=False and
the exact regularized predicted-class Mahalanobis state **once**, on KNOWN
session_01..03 only. UNKNOWN and session_04 never fit any component. Seal this
state before generating the two calibration populations. Calibrate one threshold
on non-overlapping window-7 **median** scores using the existing balanced
accuracy, lower FAR, lower FRR, higher-threshold rule. Identity voting/ties and
disabled consistency gate remain frozen. AUC/EER are diagnostic only.
**Never refit after calibration.** Record
`threshold_transport="none_same_model_calibration"`.

Fit-generation and component-identity guards accompany value-based SHA-256
fingerprints for selected indices, selector, scaler, SVM, Mahalanobis and full
fitted state. Before/after-calibration fingerprints must match. Canonical array
content hashing avoids depending on memory aliases/order during joblib
round-trips. Structural validation checks dimensions, parameters, class mapping
and learned-state fingerprints without fitting or inference.

```powershell
python -m face_recognition.presentation.cli.train_v3_deployable_model `
  --processed-root "..\face-recognition-data\v3\pose_processed" `
  --frozen-config "..\face-recognition-data\v3\models\open_set_config_v3.json" `
  --output-dir "..\face-recognition-data\v3\models\open_set_model_v3"
```

Use a fresh versioned output directory, rather than the existing `models/`
parent. There is no force option. Staging, artifact round-trip validation and
completion-manifest-last publication precede atomic directory rename; failed
construction publishes no partial final bundle. The layout is:

```text
open_set_model_v3/
  model.joblib          # selector, scaler, SVM, Mahalanobis, threshold, classes, frozen policy
  frozen_config.json    # exact source configuration bytes
  manifest.json         # aggregate provenance/counts/calibration metrics and integrity hashes
```

The joblib contains required fitted statistics/support vectors, but no retained
training/calibration frame matrix or raw landmarks. These artifacts are
biometric-derived, local-only and must never be committed. Repository-local
publication is rejected; `.gitignore` also excludes joblib/NPZ artifacts. The
loader is for trusted local bundles: hashes establish integrity, not authenticity.

```python
from face_recognition.infrastructure.persistence.v3_deployable_model import load_v3_deployable_bundle

recognizer = load_v3_deployable_bundle(model_directory)
result = recognizer.predict_window(pose_features)  # one coherent (7, 1434) pose-feature window
# result.decision: KNOWN/UNKNOWN; result.identity: registered ID or None
# result.knownness_score: median Mahalanobis score; result.threshold: persisted threshold
```

The API uses the persisted threshold unchanged and suppresses arbitrary SVM IDs
on UNKNOWN. There is no single-frame threshold API, camera integration or live
buffer in this stage. The real calibration captures should produce 260 KNOWN
and 416 UNKNOWN windows, discarding 780 and 1248 trailing frames respectively;
all 7800 fit frames are still used for fitting.

**CALIBRATION METRICS are not independent final test or generalization metrics.**
This constructs a deployable model from frozen V3 development decisions; it does
not perform a new evaluation. No additional untouched V3 holdout exists, and V2
remains the last independent external holdout. The cross-validated
pose-representation study remains the appropriate V3 development evidence.
Residual yaw sensitivity, unimproved pitch_up, unchanged pitch_down,
session/context effects, harder UNKNOWN identities and MediaPipe relative-depth
limitations remain as documented above. Calibration rates are not reported here;
the current live bundle's persisted threshold is described below.

## Live frozen V3 open-set recognition

Both platforms share the same path:
`Camera → asynchronous Face Landmarker → PoseLandmarkNormalizer → 1434 XYZ
features → loaded V3 selector/scaler/SVM/Mahalanobis → seven-observation decision
→ overlay`. Only acquisition differs: the existing OpenCV adapter on laptop,
or Picamera2 main preview plus smaller YUV inference stream on Raspberry Pi.
The deployable bundle is loaded once before opening a camera. Runtime never
reads datasets or experiment reports, fits a component, or calibrates a threshold.

Startup retains full manifest, artifact SHA-256, frozen configuration/runtime,
fitted-structure and learned-state fingerprint validation, including consistency
with the calibrated state. Live observations use lightweight fit-generation,
component-identity and frozen-contract guards instead of rehashing the selector,
scaler, SVM and large Mahalanobis matrices before and after every inference.
This avoids unnecessary Raspberry Pi CPU work without changing the frozen model.
These runtime guards detect component replacement/refitting through the runner,
not arbitrary in-place edits to learned arrays. An explicit
`recognizer.verify_integrity()` performs the expensive value-based check when
needed outside the live loop. Model bundles and publication checks are unchanged.

The model uses `all_xyz_pose_v2`, not the older roll-only normalization. Its
threshold is authoritative in the bundle (currently approximately
**-18.1917360542**), with `score = -Mahalanobis distance`; higher means more
likely KNOWN. Accept when the **window median >= persisted threshold**. These
values are distances/scores, not probabilities or confidence percentages.
There are no model/threshold/window overrides.

Accumulate **seven successful face inferences**, not seven camera frames.
Each landmark completion is consumed at most once. Preserve one active
MediaPipe inference, a latest-result slot, and busy-frame discard before pixel
conversion; no inference queue or sliding window is introduced. Each valid face
is scored once, retaining only candidate/decision-score evidence for plurality,
all-frame corresponding-score sums for ties, then deterministic class ordering.
Use median knownness, no consistency gate, then clear all seven observations.
No identity is emitted from an individual frame. The last completed label and
median remain visible as the next non-overlapping window fills.

Normal overlay: one padded/clamped face box, stable ID or `UNKNOWN`, **Last score
(completed window)**, threshold, inference FPS, last submission-to-V3 latency,
`Window: x/7`, and camera-read FPS. Before the first decision it shows
`COLLECTING` for a valid face or `NO FACE` otherwise. An UNKNOWN decision never
displays its arbitrary SVM candidate as the recognized identity. `--debug` adds
the current frame candidate/score, final temporal candidate/decision, successful
V3 completion count, MediaPipe callback count and busy skips. No fake probability
or percentage is displayed in either mode.

Debug also measures the last successful observation's **Pose+features** and
**V3 observation** in milliseconds. The latter covers the existing unchanged
selector/scaler/SVM/Mahalanobis scoring path and observation validation, before
temporal aggregation. These coarse timings use no per-frame logging or model
hashing and are disabled in normal mode. They do not change MediaPipe callback
timing or the end-to-end latency definition below.

**No-face policy:** an observed no-face result or invalid pose geometry clears
the partial window immediately. After `--absence-timeout` seconds without a
successful face observation (default **2 seconds**), clear the last decision,
score, debug candidate and box to `NO FACE`. A long gap before a new valid result
also clears any partial window; unseen callbacks already older than the timeout
are discarded. Busy/repeated snapshots never advance a window. Retain a completed
decision during a short absence grace period. This is not identity tracking:
undetected rapid person swaps cannot be associated. The landmarker requests
`num_faces=1`; if an adapter unexpectedly returns more faces, only its first is
processed deterministically. No multi-face association is performed.

Metrics have distinct meanings:

- **Camera FPS (reads):** frames acquired by the main loop; not the native sensor
  or DRM preview refresh rate, which may be higher.
- **Inference FPS:** successfully consumed pose+V3 face results per elapsed
  second; excludes no-face callbacks, dropped/busy frames and reused snapshots.
- **Latency:** from the existing monotonic MediaPipe submission timestamp through
  callback delivery, main-loop consumption, pose features, V3 and temporal work.
  It is the last successful path's latency, not camera exposure time.

Laptop (PowerShell, repository root):

```powershell
conda activate face-recognition
$env:PYTHONPATH="src"
python -m face_recognition.presentation.cli.live_open_set_recognition `
  --camera-backend opencv `
  --camera-index 0 `
  --model-dir "..\face-recognition-data\v3\models\open_set_model_v3"
```

Laptop capture keeps the device's native size/rate unless both
`--preview-width` and `--preview-height` are supplied. Shared MediaPipe inference
defaults to 480×270 and can use explicit `--inference-width/--inference-height`.
Reuse the capture protocol's camera orientation via `--orientation` when needed.
OpenCV windows exit with **q, ESC or Ctrl+C**.

Raspberry Pi (activated project environment, repository root):

```bash
PYTHONPATH=src python -m face_recognition.presentation.cli.live_open_set_recognition \
  --camera-backend picamera2 \
  --model-dir "../face-recognition-data/v3/models/open_set_model_v3"
```

Picamera2 defaults to native DRM main preview at 848×480 with 480×270 inference,
preserving the existing shared-crop strategy. DRM needs a suitable local display
session and exits with **Ctrl+C**. Explicit `--display opencv` supports a desktop
window instead. Add `--debug` to either command. Cameras, MediaPipe and GUI/native
preview resources are closed on exit and failures. Picamera2/libcamera imports
are isolated and are not required on Windows/OpenCV.

Copy the **entire three-file model bundle** (`model.joblib`, `frozen_config.json`,
`manifest.json`) and the MediaPipe `face_landmarker.task` asset to the Pi; the
asset defaults to `assets/face_landmarker.task` or an explicit
`--landmarker-model` path. Deploy the same source package/branch. No processed/raw
dataset, protocol or development report is required for live inference.
Use the project's pinned runtime requirements and Raspberry system
Picamera2/libcamera packages. The bundle loader requires matching **Python
major.minor** versions; patch releases may differ (for example, 3.11.17 and
3.11.2 are compatible). Python metadata must be a numeric `major.minor.patch`
release. **NumPy, SciPy, scikit-learn and every other recorded dependency still
require exact versions** from `manifest.json`'s creation metadata. Install
compatible builds for each platform's architecture. Frozen pose
normalizer/Mahalanobis source fingerprints must also match, including line
endings. Startup reports version/source mismatches and provides no bypass or
automatic reconstruction/retraining. Preserve the trusted local bundle unchanged.

This integration changes no experimental decision or calibration. Attendance,
names, databases, history, rosters and recognition reports remain out of scope.
