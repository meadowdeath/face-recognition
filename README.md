# Face Recognition

Layered Python project for facial landmark detection now and experimental identity recognition later. Development starts on a Windows x86_64 laptop; the intended later target is a Raspberry Pi 3B+ (Linux ARM64/aarch64) with a Raspberry Pi Camera Module 3.

**Current milestone:** OpenCV webcam or Picamera2 camera → asynchronous MediaPipe Tasks Face Landmarker → OpenCV window, native DRM preview, or console benchmark metrics. Press `q` in the OpenCV window or Ctrl+C to quit. Face detection and landmarks do not identify a person.

## Stack and architecture

**Python 3.11 is the only currently supported Python version** (`>=3.11,<3.12`). The dependencies are NumPy 1.26.4, opencv-contrib-python 4.10.0.84, MediaPipe 0.10.14, protobuf 4.25.9, and scikit-learn 1.7.2. The same versions are pinned in `pyproject.toml` and `requirements.txt`; `requirements-rpi.txt` includes those shared requirements.

The reference Raspberry Pi environment is a Raspberry Pi 3B+ running Debian 12 Bookworm on aarch64 with Python 3.11.2. It supplies Picamera2 0.3.31, libcamera, and python3-libcamera through system packages. Windows uses the shared Python dependencies without Picamera2. Scikit-learn is used by offline baseline evaluation; the production preview does not import it. The new offline dependency has not been validated on Pi hardware.

```text
src/face_recognition/
  domain/          Models and camera, detector, extractor, classifier contracts
  application/     Frame orchestration, dataset capture/processing, validation, development folds/evaluation
  infrastructure/  OpenCV and optional Picamera2 cameras, MediaPipe detector,
                   normalization/baseline features, JSONL datasets, batch baselines; live classifier/model placeholders
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

For the current session policy, this PowerShell command prompts for your actual pseudonymous ID lists without assuming their membership:

```powershell
$knownIds = @((Read-Host "Known IDs, comma-separated").Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$unknownIds = @((Read-Host "Unknown IDs, comma-separated").Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$splitArgs = @(
  "--processed-root", "..\face-recognition-data\data\processed",
  "--output", "..\face-recognition-data\splits\identity_split_v1.json",
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

Only `known_train_pool` supplies development features. For the current protocol:

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
  --split-manifest "..\face-recognition-data\splits\identity_split_v1.json"
```

This diagnostics command requires exactly two training sessions for automatic opposite folds. It prints fit/validation identity and sample counts only; exit codes are 0 success, 1 input/loading/fold failure, and 2 usage error. This command performs no feature selection, learned preprocessing, classifier fitting or evaluation.

## Full-XYZ closed-set development baseline

The offline baseline uses **all `all_xyz_v1` coordinates in stored order**: 478 landmarks yield 1,434 features; code derives the dimension rather than hardcoding it. It reuses the two opposite training-session folds above and reads only `known_train_pool` payloads. `known_test` and `unknown_pool` remain untouched. This baseline exists for later comparison against dimensionality reduction, not final holdout evaluation or live recognition.

Fixed configurations (no tuning):

- **KNN:** 5 neighbors, uniform weights, Euclidean distance, brute-force search, `n_jobs=1`.
- **SVM:** RBF, `C=1.0`, `gamma="scale"`, probability estimation disabled, 128 MB cache, `random_state=42`.
- **Random Forest:** 100 trees, Gini, unlimited depth, minimum split/leaf sizes 2/1, `max_features="sqrt"`, bootstrap enabled, no class weighting, `random_state=42`, `n_jobs=1`.

Other estimator defaults follow pinned scikit-learn 1.7.2. KNN and SVM use fresh `Pipeline(StandardScaler(copy=True, with_mean=True, with_std=True), classifier)` instances per fold. Scalers fit **only on fit rows**; validation is transformed with those fitted parameters. Random Forest has no scaler. Future trainable feature transforms must be inserted inside that same pipeline before scaling; none is implemented. At least two known identities and five fit samples are required; KNN's fixed neighbor count is never silently reduced.

Models run sequentially, with native BLAS/OpenMP threads limited to one through scikit-learn's required `threadpoolctl` dependency. Input remains one read-only `float32` matrix with fold views. Scaling necessarily allocates working matrices; SVM may internally convert to double precision. No fitted models, predictions or feature vectors are persisted.

```powershell
python -m face_recognition.presentation.cli.evaluate_baseline_models `
  --processed-root "..\face-recognition-data\data\processed" `
  --split-manifest "..\face-recognition-data\splits\identity_split_v1.json" `
  --report "..\face-recognition-data\reports\baseline_identity_v1.json"
```

Each classifier/fold reports accuracy, macro precision/recall/F1 (`zero_division=0`), and a confusion matrix with sorted pseudonymous labels (rows true, columns predicted). Summaries use **unweighted fold means and population standard deviations**; matrices stay per fold. `perf_counter` measures fit (scaler plus model) and batch prediction (including validation transformation), excluding metric computation. Mean prediction latency is batch prediction time divided by validation sample count; it is not standalone streaming latency. Timings are descriptive.

The required JSON report records these aggregates, fixed configurations, session/sample/feature counts, label order, protocol, and library versions. It contains no vectors, coordinates, individual sample predictions or absolute source paths. The report must be new, external to the repository, and outside processed/split inputs; publication is atomic without overwrite. CLI output is aggregate-only; exit codes are 0 success, 1 evaluation/report failure or interruption, and 2 usage error. Keep reports outside Git. No real model results are documented yet.

## Future work

`LandmarkNormalizer(image_width, image_height).normalize(landmarks)` provides standalone, hardware-free identity normalization. It first corrects MediaPipe's image-normalized coordinates to width-relative units `(x, y * image_height / image_width, z)`, then applies eye-centered translation, roll correction and interocular scaling. Eye centers use corners 33/133 and 362/263; z is centered/scaled only. Tests use synthetic pixel-space geometry encoded for non-square images, including 480×270. It is used by the separate PC experiment, not the production preview or feature extractor; its eye-relative output is distinct from the input image-normalized representation.

`LandmarkFeatureExtractor` flattens already identity-normalized points as `(x0, y0, z0, x1, y1, z1, ...)`, with three finite features per landmark. Capture does not invoke it.

Remaining work: compare reduced representations against the full-XYZ baseline; derive geometric features; perform final holdout evaluation and unknown rejection; validate dataset capture on Pi hardware; benchmark deployment accuracy and throughput. These are not implemented yet.
