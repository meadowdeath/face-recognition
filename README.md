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

Other estimator defaults follow pinned scikit-learn 1.7.2. KNN and SVM use fresh `Pipeline(StandardScaler(copy=True, with_mean=True, with_std=True), classifier)` instances per fold. Scalers fit **only on fit rows**; validation is transformed with those fitted parameters. Random Forest has no scaler in the baseline. At least two known identities and five fit samples are required; KNN's fixed neighbor count is never silently reduced.

Models run sequentially, with native BLAS/OpenMP threads limited to one through scikit-learn's required `threadpoolctl` dependency. Input remains one read-only `float32` matrix with fold views. Scaling necessarily allocates working matrices; SVM may internally convert to double precision. No fitted models, predictions or feature vectors are persisted.

```powershell
python -m face_recognition.presentation.cli.evaluate_baseline_models `
  --processed-root "..\face-recognition-data\data\processed" `
  --split-manifest "..\face-recognition-data\splits\identity_split_v1.json" `
  --report "..\face-recognition-data\reports\baseline_identity_v1.json"
```

Each classifier/fold reports accuracy, macro precision/recall/F1 (`zero_division=0`), and a confusion matrix with sorted pseudonymous labels (rows true, columns predicted). Summaries use **unweighted fold means and population standard deviations**; matrices stay per fold. `perf_counter` measures fit (scaler plus model) and batch prediction (including validation transformation), excluding metric computation. Mean prediction latency is batch prediction time divided by validation sample count; it is not standalone streaming latency. Timings are descriptive.

The required JSON report records these aggregates, fixed configurations, session/sample/feature counts, label order, protocol, and library versions. It contains no vectors, coordinates, individual sample predictions or absolute source paths. The report must be new, external to the repository, and outside processed/split inputs; publication is atomic without overwrite. CLI output is aggregate-only; exit codes are 0 success, 1 evaluation/report failure or interruption, and 2 usage error. Keep reports outside Git.

## Leakage-safe feature reduction evaluation

Retain the complete 1,434-coordinate `all_xyz_v1` baseline as the reference for assessing accuracy and complexity tradeoffs. The reduction experiment reruns matching classifier baselines with exactly the configurations above, then evaluates:

- **ANOVA:** `SelectKBest(f_classif)`, retaining 256, 512, 768 or 1024 individual coordinates. No landmark grouping or correlation pruning is performed.
- **PCA:** 90%, 95% or 99% explained variance, using full SVD, `copy=True`, and no whitening. The actual component count is reported independently for each fold.

Every selector, scaler and PCA is fitted inside a fresh per-fold pipeline using only fit samples (and fit labels for ANOVA). Validation is only transformed. Fitting reduction globally before creating folds would leak validation information into feature selection or the learned coordinate system. **`known_test` and `unknown_pool` payloads remain untouched.**

| Representation | KNN / SVM | Random Forest |
|---|---|---|
| Baseline | StandardScaler → classifier | classifier |
| ANOVA | SelectKBest → StandardScaler → classifier | SelectKBest → classifier |
| PCA | StandardScaler → PCA → classifier | StandardScaler → PCA → classifier |

There is no second scaler after PCA. Selected ANOVA indices retain source-coordinate order. PCA components are not individual landmark coordinates. Source dimensions remain generic; an ANOVA count exceeding the source width fails explicitly, rather than being clipped or skipped.

```powershell
python -m face_recognition.presentation.cli.evaluate_feature_reduction `
  --processed-root "..\face-recognition-data\data\processed" `
  --split-manifest "..\face-recognition-data\splits\identity_split_v1.json" `
  --report "..\face-recognition-data\reports\feature_reduction_v1.json"
```

The default grid is 24 representation/classifier configurations and 48 fold fits, executed sequentially with the existing single-thread resource policy. Each fit creates and releases its own working pipeline; no transformed-dataset cache is retained. Full SVD is deterministic within numerical tolerance and can make this experiment substantially slower than the baseline. Input stays `float32` where compatible; temporary scaling/SVD/model arrays are unavoidable.

The external, atomic, no-overwrite JSON report includes per-fold metrics, confusion matrices, timings, actual dimensions, optional technical ANOVA indices, aggregate means/population standard deviations, and comparisons to the matching classifier baseline **from this run**. It contains no scores, component vectors, features, landmarks or absolute source paths. Fit/prediction timing includes all preprocessing; latency remains a batch average.

The primary metric is **mean macro F1**, followed by mean accuracy, fewer **mean retained dimensions across folds**, and lower mean prediction latency. Remaining ties use reduction ID and classifier name. Ranking is deterministic for recorded measurements; close timing ties can change order across runs. Metric/latency deltas are candidate minus baseline (positive latency delta means slower); feature reduction is original dimension minus mean retained dimension, with percentage relative to the original width. All per-fold dimensions remain available when PCA counts differ. There are no pass/fail thresholds and **no final model is selected**.

For small synthetic experiments, repeat `--anova-k` and/or `--pca-variance` to explicitly replace that family's defaults (fractions such as `0.95`, not `95`). The full baseline is always included. This reduction command performs no classifier tuning, correlation analysis or grouped landmark selection.

## Focused ANOVA + SVM development tuning

The preceding development experiment identified **ANOVA-512 + RBF SVM** as the strongest tested configuration: mean accuracy **0.810385**, mean macro F1 **0.801956**, with fold F1 values **0.849215 / 0.754697**. The full-feature SVM reference had accuracy **0.773269** and macro F1 **0.770397**. These session-aware development observations motivate a focused search; they do not establish a final model. PCA is excluded from this stage.

The tuning command evaluates exactly **60 configurations**, with no grid overrides:

- ANOVA `k`: **384, 448, 512, 640, 768**.
- SVM `C`: **0.5, 1.0, 2.0, 4.0**.
- SVM `gamma`: **"scale", 0.001, 0.01**.
- RBF kernel only, `probability=False`; all other SVM/scaler settings are inherited from the existing baseline.

Each configuration uses both existing opposite-session folds inside `known_train_pool`. A fresh **SelectKBest(f_classif) → StandardScaler → SVM** pipeline fits exclusively on that fold's fit rows/labels. Validation is transformed and predicted only. **`known_test` and `unknown_pool` feature payloads remain untouched**. There is no random frame/sample split, probability calibration, final all-development-data fit, or model persistence.

```powershell
python -m face_recognition.presentation.cli.tune_anova_svm `
  --processed-root "..\face-recognition-data\data\processed" `
  --split-manifest "..\face-recognition-data\splits\identity_split_v1.json" `
  --report "..\face-recognition-data\reports\anova_svm_tuning_v1.json"
```

The search runs **120 fold fits sequentially**, plus **two full-feature SVM reference fits**. It reuses the float32 source matrix, single-native-thread limit, timers, metrics and population-standard-deviation aggregation. Working models/matrices are released between runs. The fixed grid requires at least 768 source features and at least two known identities; invalid input fails before fitting.

Rank by mean macro F1 first, then higher worst-fold macro F1, smaller absolute fold F1 gap, higher mean accuracy, fewer features, and lower mean batch prediction latency. Final ties use C and configured gamma order. Timings are descriptive, so timing ties can reorder across runs. The winning row is called a **selected development candidate**; unknown-rejection and final holdout stages are still outstanding.

For each k, report shared selected indices across the two folds, overlap `100 * shared / k`, and Jaccard `shared / union`. Selections must remain identical for a given k/fold across C/gamma; otherwise the run fails. Default CLI output shows aggregate overlap and the top ten candidates, not indices or confusion matrices. SVM diagnostics record total/per-class support-vector **counts**, aligned with the deterministic pseudonymous label order; support-vector values are never serialized.

The external aggregate JSON report contains all 60 candidates, both folds' metrics/confusion matrices/timings/counts, overlap diagnostics and technical indices, ranking, selected development candidate, search space and library versions. It refuses existing destinations and repository/input-tree locations, using the shared atomic report writer. No feature/support-vector values or absolute source paths are stored. Exit codes are 0 success, 1 failure/interruption, and 2 usage error.

Comparisons use the **full-feature SVM freshly evaluated in this run** and the exact **k=512/C=1/gamma="scale"** grid row. No historical score or timing is hardcoded into the evaluator. Metric/latency deltas are candidate minus reference; feature reduction is reference dimension minus k (negative means more features), with percentage relative to that reference dimension. No real tuning results are documented yet.

## Development UNKNOWN rejection calibration

The surviving configurations are **ANOVA-512 / RBF SVM C=4, gamma=0.001** and **ANOVA-448 / RBF SVM C=4, gamma=0.001**, both with `probability=False`. This stage compares their ability to reject unseen identities; it does not reopen tuning or train a final recognizer.

Unknown identities must be separated **by whole identity**, with all their sessions inheriting one role. Reusing calibration identities for final unknown evaluation would expose the threshold-selection procedure to the same people used to assess it. Assign roles explicitly **before inspecting rejection performance**. No role is inferred from session counts.

`create_unknown_rejection_split` accepts `--identity-split`, a new external `--output`, and repeatable `--calibration-id` / `--holdout-id` arguments. Both lists are required, disjoint, and together must cover the source `unknown_pool` exactly. The logical manifest contains counts and pseudonymous IDs, not copied features. A canonical metadata-only identity-split digest binds it to its source; no absolute paths or feature-data fingerprints are stored. Existing files are never overwritten.

Verify an existing unknown split and print counts **without opening any feature files**:

```powershell
python -m face_recognition.presentation.cli.create_unknown_rejection_split `
  --identity-split "..\face-recognition-data\splits\identity_split_v1.json" `
  --verify "..\face-recognition-data\splits\unknown_rejection_split_v1.json"
```

Only the new calibration loader may open `unknown_calibration`; earlier development commands continue to protect the entire unknown pool. **`known_test` and `unknown_holdout` feature payloads remain untouched**. Source file catalogs and split metadata may be checked without reading protected payloads.

```powershell
python -m face_recognition.presentation.cli.calibrate_unknown_rejection `
  --processed-root "..\face-recognition-data\data\processed" `
  --identity-split "..\face-recognition-data\splits\identity_split_v1.json" `
  --unknown-split "..\face-recognition-data\splits\unknown_rejection_split_v1.json" `
  --report "..\face-recognition-data\reports\unknown_rejection_calibration_v1.json"
```

Each candidate performs two sequential, independent **known-fit-only** ANOVA → scaler → SVM fits. That fold's known out-of-session validation samples and the authorized unknown calibration samples are then scored by the fitted pipeline. Unknowns never reach any fitting step. Their feature matrix is reused read-only; no fitted models, decision vectors or per-sample confidences are persisted.

The single confidence definition is **highest minus second-highest SVM OVR decision score**, in deterministic class order. These are margins, not probabilities. Existing SVM `predict` behavior supplies identity predictions, including its existing vote-tie behavior; no classifier parameters are changed. For binary SVC only, the signed decision value `d` is represented as class scores `[-d, d]`, giving the same top-two margin rule. Nonfinite scores and fewer than two class columns are rejected.

Accept the predicted known identity when `margin >= threshold`; otherwise reject as `UNKNOWN`. Each fold searches sorted unique observed margins, valid interior midpoints, zero (accept all), and the next representable finite value above the maximum (reject all). It maximizes **binary balanced accuracy**, then prefers lower unknown **FAR** (fraction of unknown samples accepted), lower known **FRR** (fraction of known samples rejected), and higher threshold. Integer count comparisons avoid rounding-based false ties. Binary macro F1 is also reported; binary acceptance does not imply the identity was correct.

Open-set diagnostics explicitly distinguish accepted-correct known samples, accepted-misidentified known samples, rejected known samples, rejected unknowns, and falsely accepted unknowns. Exact open-set accuracy counts only accepted-correct known samples plus rejected unknowns. Reports retain closed-set macro metrics as a separate reference.

For each candidate, the **development rejection threshold** is the median of the two fold thresholds: after sorting `a <= b`, use `a + (b-a)/2`. The report includes their mean, median and absolute difference. Candidate ranking uses unweighted means at the fold-specific optimum thresholds: higher exact open-set accuracy, lower unknown FAR, higher known correct-identification rate, higher binary balanced accuracy, smaller threshold difference, then fewer features. Metrics at the transferred midpoint are stored separately for transparency. The same unknown calibration identities are scored in both folds; this is not additional independent data.

The external aggregate JSON report contains split metadata, both candidates, thresholds, rates, means/population standard deviations, threshold stability, ranking, and the **selected open-set development candidate**. It includes no feature values, per-sample confidence/decision vectors or absolute paths, and uses atomic no-overwrite publication outside repository/input trees. CLI output is aggregate-only; exit codes are 0 success, 1 failure/interruption and 2 usage error. No real UNKNOWN calibration results are documented yet.

**These are calibration-set development diagnostics, not independent evaluation results.** The SVM's absolute margin distribution may shift when refitted using more known training data. A later holdout stage must measure whether this median threshold transfers adequately. No final production model is selected or persisted, and the reserved holdouts have not been evaluated.

## Comparing UNKNOWN rejection scores

The first margin-only experiment had FAR about **0.357 / 0.403** and FRR about **0.356 / 0.334** for ANOVA-512 / ANOVA-448, despite closed-set macro F1 around **0.817** for both. Stable fold thresholds did not imply adequate known/unknown separation. Closed-set classification chooses a known label; open-set rejection must additionally recognize when none of those identities is appropriate.

This controlled comparison leaves the two **C=4, gamma=0.001, RBF, probability=False** identity classifiers unchanged. There are exactly four confidence definitions, all oriented so **larger means more likely KNOWN**:

| Score family | Definition |
|---|---|
| `svm_margin` | Highest minus second-highest OVR decision score; retained as the reference |
| `svm_top1` | Maximum OVR decision score |
| `centroid_distance` | Negative Euclidean distance to the **SVM-predicted class** centroid in ANOVA-selected/scaled space |
| `normalized_centroid_distance` | Negative predicted-class centroid distance divided by that class's median fit distance |

Centroids are means of **known fold-fit samples only**, grouped by true fit labels. Class distance scales are medians of those same samples' distances to their own centroid, floored at a fixed **1e-8** in selected/scaled Euclidean units. Validation and unknown calibration data never affect centroids, scales, ANOVA, StandardScaler or SVM fitting. One fitted classifier per candidate/fold is shared by all four scores: **four identity-model fits total**, sequentially, with no transformed-dataset cache. Existing SVM identity predictions, class order and binary `[-d,d]` score convention are preserved.

```powershell
python -m face_recognition.presentation.cli.evaluate_rejection_scores `
  --processed-root "..\face-recognition-data\data\processed" `
  --identity-split "..\face-recognition-data\splits\identity_split_v1.json" `
  --unknown-split "..\face-recognition-data\splits\unknown_rejection_split_v1.json" `
  --report "..\face-recognition-data\reports\unknown_rejection_scores_v1.json"
```

This command uses only known development folds and the existing explicit `unknown_calibration` identities. **`known_test` and `unknown_holdout` biometric payloads remain untouched**, with separate protection flags in the report and guarded-file tests. It does not create a new unknown split, change identity roles, tune classifiers or reopen the 60-candidate grid.

Threshold-independent diagnostics include **ROC AUC with KNOWN positive**, calculated from continuous scores (probability a known score outranks an unknown score, counting ties by half), and discrete **EER**. EER uses the finite observed-score/midpoint/boundary operating point with FAR and FRR closest; ties prefer lower FAR, lower FRR, then higher threshold. Reported EER is `(FAR+FRR)/2` there, with no interpolation. The original balanced-accuracy threshold search and its FAR/FRR/higher-threshold tie rules remain unchanged; signed thresholds are enabled for negative-distance scores.

Diagnostic operating points target **FAR <=20%, <=10%, <=5%**. Under each cap, choose maximum known acceptance, then lower FAR, then higher threshold. **FAR** is the fraction of unknown samples accepted; **FRR** is the fraction of known samples rejected. Points accepting **at most 1% of known samples** are explicitly flagged `impractical` (the documented meaning of "effectively all rejected"), while retaining their thresholds and measured rates. This flag and the FAR caps are diagnostic conventions, **not application requirements**. A point with no finite feasible threshold is reported unavailable.

Per fold, report AUC, EER and its threshold, the balanced-accuracy calibration point, FAR-constrained points, and separate known/unknown score-distribution count, mean, population std, median, and linearly interpolated 10th/25th/75th/90th percentiles. Reports aggregate both folds, retain AUC/FAR/FRR/open-accuracy gaps, and include pooled score summaries. Pooled unknown counts are **score observations from two fitted models**, not additional independent subjects. Binary acceptance metrics remain separate from accepted-correct known identification, accepted-known misidentification, and exact open-set accuracy. Median-threshold diagnostics are also retained separately from fold-optimal metrics.

Rank the eight model/score combinations by higher mean AUC, lower mean EER, lower FAR at the balanced-accuracy threshold, higher correct-identification rate, higher exact open-set accuracy, smaller absolute threshold gap, then fewer ANOVA features. Final ties use score-family name. Threshold gaps are in each score's own units. Deltas compare each score with the **same classifier's margin reference from these identical fits**. The first-ranked result is only the **best rejection-score development candidate**.

The external aggregate JSON report uses atomic no-overwrite publication and contains no per-sample scores, feature/support vectors, centroid vectors, class scale vectors or absolute source paths. The CLI prints aggregate diagnostics only. Calibration-set metrics are not independent holdout estimates; any threshold transfer after refitting still requires later holdout evaluation. No real score-comparison results are documented yet, and no final recognizer is selected or persisted.

## Temporal open-set development evaluation

Single-frame rejection did not separate known and unknown samples adequately: the best distance-score development result still rejected many known observations. Attendance decisions should therefore be evaluated across several observations rather than relying on one inference. This experiment keeps **ANOVA-448 → StandardScaler → RBF SVM C=4, gamma=0.001, probability=False** fixed and compares only **centroid_distance** and the **svm_margin** reference. There is no new classifier, score family or model tuning.

Use **non-overlapping windows of 1, 3, 5 and 7 samples**. Within each person/session/condition, sort by `sample_index`, anchor at the smallest index, and discard only trailing incomplete windows. Duplicate or missing indices fail explicitly. Windows never cross an identity, session or condition. Overlapping windows are avoided because shared samples would inflate the apparent observation count. These time-spaced, static-condition captures approximate temporal observations; they do not measure continuous-video performance or end-to-end decision latency. A later Raspberry implementation may use a rolling live window, but none is introduced here.

Identity aggregation uses the largest vote count. Ties use the highest sum of that class's SVM decision scores **across every frame**, then deterministic class order for an exact remaining tie. Confidence is the **median of every frame's confidence**, including frames voting for another identity; larger always means more likely KNOWN. Vote diagnostics distinguish unanimous windows, a unique but non-unanimous plurality (the reported majority category), and tied top vote counts before tie breaking.

There are only **two sequential identity-model fits**, one per known development fold. ANOVA, scaling, SVM and centroids use known fold-fit data only. Each fold/score/window size independently calibrates its threshold using known out-of-session validation windows and `unknown_calibration` windows, reusing the existing balanced-accuracy objective and FAR/FRR/higher-threshold tie rules. Frame thresholds are never reused for larger windows. **`known_test` and `unknown_holdout` biometric payloads remain untouched.**

```powershell
python -m face_recognition.presentation.cli.evaluate_temporal_open_set `
  --processed-root "..\face-recognition-data\data\processed" `
  --identity-split "..\face-recognition-data\splits\identity_split_v1.json" `
  --unknown-split "..\face-recognition-data\splits\unknown_rejection_split_v1.json" `
  --report "..\face-recognition-data\reports\temporal_open_set_v1.json"
```

The external atomic, no-overwrite JSON report contains ROC AUC, discrete EER and threshold, balanced-accuracy calibration, FAR/FRR and identification metrics, 20%/10%/5% FAR diagnostics, and used/discarded/window counts. It retains fold-specific thresholds, their midpoint development threshold, and separate metrics at that midpoint. Means are unweighted across folds, with population standard deviations; count totals are **fold observations**, so unknown windows scored by both models are counted twice. Condition diagnostics aggregate across identities using each fold's global threshold, without condition-specific calibration or participant-level results. No per-frame scores, vote sequences, feature/centroid vectors or absolute paths are stored.

Every row includes deltas against the **same score's window-size-1 reference**. Rank by higher mean AUC, lower EER, lower FAR, higher correct known identification, higher exact open-set accuracy, lower FRR, then smaller window; final ties use score name. Larger windows are not automatically better: inspect improvements and discarded samples before choosing decision latency. The first-ranked row is only the **best temporal open-set development candidate**. These are calibration-set diagnostics, not independent holdout estimates; no final model is selected or persisted, and no real temporal results are documented yet.

## Future work

`LandmarkNormalizer(image_width, image_height).normalize(landmarks)` provides standalone, hardware-free identity normalization. It first corrects MediaPipe's image-normalized coordinates to width-relative units `(x, y * image_height / image_width, z)`, then applies eye-centered translation, roll correction and interocular scaling. Eye centers use corners 33/133 and 362/263; z is centered/scaled only. Tests use synthetic pixel-space geometry encoded for non-square images, including 480×270. It is used by the separate PC experiment, not the production preview or feature extractor; its eye-relative output is distinct from the input image-normalized representation.

`LandmarkFeatureExtractor` flattens already identity-normalized points as `(x0, y0, z0, x1, y1, z1, ...)`, with three finite features per landmark. Capture does not invoke it.

Remaining work: evaluate rejection-threshold transfer and reserved holdouts; consider correlation/grouped landmark selection and geometric features; validate dataset capture on Pi hardware; benchmark deployment accuracy and throughput. These are not implemented yet.
