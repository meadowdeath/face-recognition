# Face Recognition

## Attendance kiosk interface — Phase 1

The optional PyQt5 Widgets interface runs independently of camera, model and
dataset assets. It provides nine persistent screens: standby, attendance,
demonstration PIN access, administrator dashboard, sessions, records, export,
system status and settings. All displayed records and diagnostics are fictional.
The demonstration PIN is **123456**; this is navigation scaffolding, not security.

Windows development (native Qt backend, 800 × 480 window):

```powershell
python -m pip install -e ".[gui]"
$env:PYTHONPATH="src"
python -m face_recognition.presentation.gui --windowed
```

The GUI extra pins the PyQt5 binding to 5.15.11 for desktop development; it is
not a recognition-core dependency. On Raspberry Pi use the already installed
system PyQt5 / Qt 5.15.8 through a `--system-site-packages` environment. Do not
replace the working system Qt/EGLFS installation with desktop wheels.

```bash
QT_QPA_PLATFORM=eglfs PYTHONPATH=src python -m face_recognition.presentation.gui
```

EGLFS launches fullscreen; the application does not set the platform backend.
Escape closes the development application. Dark Amethyst is the default theme;
Settings → Appearance (`Ajustes → Apariencia`) switches all existing widgets to
Light Lavender immediately. The choice lives only in memory. One timer updates
the date/time once per second. No inactivity timer, hardware polling or worker
is started.

Attendance's visual-state selector and export's USB selector demonstrate the
required states. Session, export, PIN-change, brightness, clock and power
controls are visual placeholders with no operational actions. There is no real
authentication, persistence, export, camera preview or recognition integration.
Touchscreen/stylus deployment and integrated Raspberry performance remain
future work; static offscreen rendering does not validate integrated throughput.

Optional offscreen checks (skip automatically if PyQt5 is absent):

```powershell
$env:QT_QPA_PLATFORM="offscreen"
python -m unittest tests.unit.test_kiosk_gui
```

GUI code is contained in `presentation/gui`: `window.py` owns screens and
navigation, `themes.py` owns shared palette/style definitions, and `__main__.py`
starts the application. It imports no recognition or camera components.

Operational facial dataset capture, pose-feature processing, selected-model
construction and single-face open-set recognition on Windows/OpenCV and
Raspberry Pi/Picamera2. Attendance registration and storage are not implemented.

The completed research is preserved in Git. **`v3-open-set-model` (`755a06f`)**
preserves the frozen research/model-construction workflows; **`v2-open-set-final`**
preserves the last independently evaluated external holdout. Research commands,
CV planning, comparisons, diagnostics and discarded classifiers/scorers are no
longer executable from current HEAD.

## Installation

Python **3.11** is supported. Keep the pinned requirements; do not copy virtual
environments between Windows and Linux.

Windows PowerShell, repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python tools/download_face_landmarker.py
```

Alternatively activate the existing `face-recognition` Conda environment and set
`$env:PYTHONPATH="src"` when running without an editable installation.

Raspberry Pi 3B+ reference: Debian 12 Bookworm/aarch64, Camera Module 3, Python
3.11.2 and system Picamera2 0.3.31/libcamera. Install the distribution's
`python3-picamera2`/`python3-libcamera` packages, then:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements-rpi.txt
python -m pip install --no-deps -e .
python tools/download_face_landmarker.py
```

Picamera2/libcamera imports are isolated and lazy; Windows does not require them.
The MediaPipe asset is `assets/face_landmarker.task`, or supply `--landmarker-model`
to live recognition. Source code, requirements and trusted model assets can be
shared; interpreters and binary packages must be installed for each architecture.

## Architecture and preview

```text
Camera (OpenCV / Picamera2)
  â†’ asynchronous MediaPipe Face Landmarker (LIVE_STREAM)
  â†’ pose-relative XYZ features
  â†’ loaded ANOVA / scaler / SVM / Mahalanobis
  â†’ seven successful observations / median / deterministic identity vote
  â†’ KNOWN identity or UNKNOWN overlay
```

The existing `domain`, `application`, `infrastructure`, `presentation` and `config`
layers remain. Camera acquisition is platform-specific; recognition is shared.
There is one active MediaPipe request, a latest-result slot, and immediate busy
frame discard before conversion. No inference backlog, worker or frame queue is
introduced. Each completed landmark result is consumed at most once.

Preview and performance utilities:

```powershell
python -m face_recognition.presentation.cli.recognize --camera opencv --display opencv
```

```bash
PYTHONPATH=src python -m face_recognition.presentation.cli.recognize --camera picamera2 --display drm --orientation rotate180
PYTHONPATH=src python -m face_recognition.presentation.cli.recognize --camera picamera2 --display none --landmarks none
```

`--landmarks none|selected|all` controls drawing; selected mode requires
`--indices INDEX,INDEX,...`. Orientation supports `normal`, `rotate180`,
`mirror-horizontal`, `mirror-vertical`. Preview defaults are 848Ã—480 main and
480Ã—270 inference. Picamera2 retains the shared-crop main preview plus lores YUV
inference stream, stride-aware conversion only after reserving inference, and
native DRM preview/transparent overlay. DRM and no-display modes do not convert
camera pixels for presentation. OpenCV live capture retains device defaults
unless explicit preview dimensions are supplied.

## Guided dataset capture

Capture writes raw landmarks and session metadata, validates existing captures,
and supports explicit resume. Use pseudonymous identifiers; no participant names
or images are required. SPACE begins/advances the guided conditions; q exits.

```powershell
python -m face_recognition.presentation.cli.capture --help
python -m face_recognition.presentation.cli.capture --person-id p001 --session-id session_01 --data-root "..\face-recognition-data\data\raw"
```

Use `--camera picamera2 --display drm` on Raspberry Pi and the appropriate
`--orientation` for the physical camera. The default guided protocol covers
center, near/far, roll, yaw, pitch and four translations. Capture parameters,
complete conditions, sample order and inference dimensions remain in manifests.
Rerun the same capture command with matching session configuration to resume;
existing complete conditions are validated rather than silently overwritten. Consult `--help`
for capture timing and backend options.

## Selected pose dataset

Process raw captures into a **new external output root**:

```powershell
python -m face_recognition.presentation.cli.process_dataset `
  --raw-root "..\face-recognition-data\data\raw" `
  --output-root "..\face-recognition-data\v3\pose_processed"
```

The command emits only `all_xyz_pose_v2`: 478 landmarks Ã— XYZ = 1434 features,
with unchanged identity/session/condition/index/timestamp provenance. Completed
outputs cannot be overwritten. Failures retain failed staging metadata and never
publish a completed dataset. Historical processed datasets remain untouched.

The exact frozen transform corrects `(x, y*height/width, z)`, averages eye groups
`[33,133]` and `[362,263]`, uses their midpoint as origin and 3D separation as
scale, and builds an orthonormal right-handed frame. Horizontal points between
the eye centers; vertical is chin `[152,148,377]` minus forehead `[10,109,338]`,
Gramâ€“Schmidt projected off horizontal; depth is their cross product, followed by
vertical reconstruction. Coordinates are `((p-origin)/scale) @ basis`. Collapsed,
collinear or non-finite geometry fails safely. The normalization and Mahalanobis
source files are intentionally unchanged because frozen bundles fingerprint them.
MediaPipe z is relative depth, not metric 3D frontalization.

## Model construction and calibration

The frozen configuration is authoritative: `all_xyz_pose_v2`, 1434 inputs,
`SelectKBest(f_classif, k=448)`, `StandardScaler`, RBF `SVC(C=4, gamma=0.001,
probability=False)` and predicted-class regularized Mahalanobis. Covariance uses
the existing per-class Ledoitâ€“Wolf estimate, ridge
`1e-8*max(1, trace(covariance)/dimension)` and Cholesky solve. Knownness is negative
Mahalanobis distance. There are no method/hyperparameter overrides.

Model construction uses the frozen population metadata, not new split selection:
KNOWN sessions 01â€“03 fit every component exactly once; KNOWN session 04 and both
sessions of all eight designated UNKNOWN identities calibrate only. Membership
is read from authenticated metadata. UNKNOWN and KNOWN calibration data never
fit preprocessing, classifier or covariance statistics. Current declared real
counts are 7800 fitting frames, 2600 KNOWN calibration frames and 4160 UNKNOWN
calibration frames; actual catalog, per-condition counts and payloads are checked.

```powershell
python -m face_recognition.presentation.cli.train_recognition_model `
  --processed-root "..\face-recognition-data\v3\pose_processed" `
  --frozen-config "..\face-recognition-data\v3\models\open_set_config_v3.json" `
  --output-dir "..\face-recognition-data\v3\models\open_set_model_v3"
```

The existing bundle is already trained; cleanup and deployment do not require
running this command. An existing output directory is refused. Training retains
read-only authentication of the frozen protocol and pose-report provenance;
`--protocol` and `--pose-report` support relocated files only when hashes match.
It does not regenerate protocols, replay experiments or select strategies.

Threshold calibration uses non-overlapping, boundary-safe windows of seven and
median scores. It maximizes binary balanced accuracy with exact integer
comparisons, then lower FAR, lower FRR and higher numeric threshold. Enumeration
retains observed scores, finite interior midpoints and accept/reject boundaries.
AUC/EER are diagnostics. **No refit follows calibration**:
`threshold_transport="none_same_model_calibration"`. Component and full-state
fingerprints before and after calibration must match.

Calibration metrics are not independent test/generalization performance. No new
untouched V3 holdout exists; V2 remains the last independent external holdout.

## Model bundle and loading

```text
open_set_model_v3/
  model.joblib
  frozen_config.json
  manifest.json
```

Publication stages and round-trips artifacts, writes the completion manifest
last, verifies hashes/state and publishes the directory without overwrite.
The manifest contains aggregate calibration counts/metrics, provenance and
fingerprints. It does not contain raw features, individual scores or predictions.
The fitted selector/scaler/SVM/Mahalanobis state is biometric-derived and local.

```python
from face_recognition.infrastructure.persistence.recognition_model import load_recognition_bundle
recognizer = load_recognition_bundle(model_directory)
# features: seven finite all_xyz_pose_v2 rows, shape (7, 1434)
decision = recognizer.predict_window(features)
# decision.identity is None for UNKNOWN; score is a distance-derived value.
```

Only load trusted local joblib bundles: integrity hashes are not authenticity
signatures. Startup verifies manifest/artifact SHA-256, configuration,
implementation fingerprints, fitted structure, learned-state fingerprints and
calibration consistency before camera access. Python **major.minor** must match;
patch releases may differ. Recorded NumPy, SciPy and scikit-learn versions must
match exactly. No dependency upgrade or validation bypass is provided.

Historical internal class/module paths (`v3_open_set_config`,
`v3_deployable_model`, `sklearn_v3_novelty.FittedV3Identity`, `EvaluationVersions`)
remain where serialized objects and canonical fingerprints require them. They
contain compatibility schemas, not executable experiment workflows. Bundle/schema
identifiers, representation IDs, manifest builder IDs and frozen source hashes
are preserved; existing bundles need no migration.

Live inference uses lightweight fit-generation, component-identity and frozen
contract guards. It never rehashes/serializes large learned arrays per frame.
`recognizer.verify_integrity()` remains an explicit expensive diagnostic outside
the hot loop; lightweight guards do not detect arbitrary in-place array edits.

## Live recognition

Windows PowerShell:

```powershell
python -m face_recognition.presentation.cli.live_recognition `
  --camera-backend opencv --camera-index 0 `
  --model-dir "..\face-recognition-data\v3\models\open_set_model_v3"
```

Raspberry Pi:

```bash
PYTHONPATH=src python -m face_recognition.presentation.cli.live_recognition \
  --camera-backend picamera2 \
  --model-dir "../face-recognition-data/v3/models/open_set_model_v3"
```

Copy the entire unchanged three-file bundle, source package and MediaPipe task
asset to the Pi. No raw/processed dataset or experimental report is required for
live inference. OpenCV exits with q, ESC or Ctrl+C; native DRM exits with Ctrl+C.
Camera, detector, display and native preview resources close on errors and exit.

Seven **successful face inferences** form a non-overlapping window. Identity uses
plurality, ties by summed corresponding SVM decision scores across all seven
observations, then class order. Score uses median, without a consistency gate.
Accept `score >= persisted threshold` (current bundle: approximately
**-18.1917360542**). UNKNOWN suppresses the arbitrary SVM candidate identity.
The completed decision/score remains visible while the next window fills.

An observed no-face result or invalid pose immediately clears the partial
window. After `--absence-timeout` seconds (default 2), the last decision, score,
candidate and box clear to NO FACE. Old callbacks are rejected; a long gap before
reappearance resets partial observations. Single-face mode uses the first
returned face deterministically, without multi-face tracking or identity gates.

Normal overlay: face box, stable identity/UNKNOWN, last completed temporal score,
threshold, inference FPS, submission-to-V3 latency, window occupancy and camera
read FPS. `--debug` adds frame/temporal candidates, frame score, completions,
callback/busy counts and coarse **Pose+features / V3 observation** milliseconds.
Scores are not probabilities; there is no confidence percentage.

## Performance and tests

Camera FPS counts reads, not native DRM refresh. Live inference FPS counts
successfully consumed pose+recognition completions, excluding dropped/busy,
no-face and reused results. Live latency runs from MediaPipe submission through
callback/main-loop/pose/recognition/temporal completion. Preview inference FPS
counts callbacks, including no face. Preview timing breaks down preprocessing,
dispatch, async result and total callback time; reused results retain their timing.
Debug coarse timings do not change those callback semantics or add per-frame logs.

The retained LIVE_STREAM design previously measured about 7â€“8 active-face
MediaPipe completions/s and 114â€“120 ms callback latency on Raspberry Pi 3B+.
These are hardware observations, not an end-to-end recognition guarantee.

Hardware-free tests use synthetic fixtures and temporary directories only:

```powershell
$env:PYTHONPATH="src"
python -m unittest discover -s tests
git diff --check
```

On machines with heavily oversubscribed native numerical libraries, optionally
set `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` for testing.
No test accesses the external biometric data tree or requires camera hardware.

## Research evidence and limitations

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


The cross-validated representation comparison is iterative development evidence;
it is not an independent final holdout. V3 research/model-selection commands are
available only from the frozen historical tag. Residual yaw/pitch sensitivity,
session/context variation and harder-to-reject UNKNOWN identities remain known
limitations. No explicit lighting labels exist. Do not retune frozen model
versions from archived final results.

## Privacy

Raw landmarks, processed features, trained selectors/scalers/SVMs, covariance
references and model bundles are sensitive biometric-derived artifacts. Keep
all generated data and bundles local and outside Git. Existing `.gitignore`
protections for datasets, `models/`, `*.joblib`, `*.npz`, MediaPipe assets and
secrets remain. Do not put participant names, raw payloads, individual predictions
or scores in documentation. No attendance database, roster or recognition history
is implemented.
