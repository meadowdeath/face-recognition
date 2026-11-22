# Dynamic population training

Dynamic construction uses the selected V3 pose representation, ANOVA-448,
StandardScaler, RBF SVM, predicted-class regularized Mahalanobis distance and
seven-observation temporal policy. The frozen V3 CLI and bundles retain their
archived contracts. Dynamic training uses fresh population and dataset provenance.

A population requires at least two KNOWN identities and at least
`(known_count + 1) // 2` UNKNOWN identities. This is a minimum, not an exact
ratio. Every selected UNKNOWN identity contributes to calibration and there is
no fixed maximum. The rule is a project policy, not a FAR or accuracy guarantee.
Full retraining is required because feature selection, feature scaling, class
boundaries, and per-class covariance references depend on the fitting population.
Each run fits once and calibrates with the same fitted state. Calibration
metrics describe the calibration captures, not independent generalization.

Each selected identity uses the guided protocol of 478 landmarks, 13 ordered
conditions, 20 samples per condition, five warm-up results and a 250 ms minimum
sample interval. KNOWN uses `session_01` through `session_04` in chronological
order, with 780 fit and 260 calibration samples per identity. UNKNOWN uses
`session_01` and `session_02`, both for calibration, for 520 samples per identity.
Capture sessions independently. Preserve their original manifests and file
contents. The processing step checks declared capture metadata and source bytes.

Use only authorized, consented captures. Keep raw data, derived features and
model bundles outside Git on controlled storage. Use synthetic data in tests.
SQLite name mappings are managed separately. Registering an ID in SQLite neither
assigns it a training role nor adds it to a model.

## Example expansion from 10 to 11 KNOWN identities

This example reserves `p019` as a new KNOWN identity and retains the existing eight
UNKNOWN identities. The result is **11 KNOWN, 8 UNKNOWN, minimum 6 UNKNOWN**.
It yields 8,580 fitting samples, 2,860 KNOWN calibration samples and 4,160 UNKNOWN
calibration samples.

On the Raspberry Pi, run the capture CLI for each session at its scheduled
capture time, changing the session argument for each independent visit. Quit
the kiosk through the administrator dashboard before using the camera for capture.

```bash
cd ~/Projects/face-recognition
source .venv/bin/activate
export PYTHONPATH=src
RAW="$HOME/Projects/face-recognition-data/data/raw"

python -m face_recognition.presentation.cli.capture \
  --camera picamera2 --display drm --orientation rotate180 \
  --person-id p019 --session-id session_01 --data-root "$RAW"
# Repeat at the separate visits with session_02, session_03 and session_04.
```

On Windows, run from the repository in the existing Python environment. Enter
the actual authorized SSH destination when prompted. Copy the complete identity
directory to a fresh external incoming directory. No transfer is automatic.

```powershell
$env:PYTHONPATH = 'src'
$Pi = Read-Host 'Authorized Pi SSH destination (user@host)'
$Run = '..\face-recognition-data\retraining'
$Incoming = Join-Path $Run 'incoming_01'
New-Item -ItemType Directory -Path $Incoming -ErrorAction Stop | Out-Null
scp -r "${Pi}:~/Projects/face-recognition-data/data/raw/p019" "$Incoming"
```

Save the following as `..\face-recognition-data\retraining\population_01.json`.
These lists are explicit role assignments. Replace them with your reviewed
assignments if your captures differ. Case-insensitive ID collisions and unknown
JSON fields are rejected. The reader orders identities deterministically.

```json
{
  "schema_version": 1,
  "population_policy": "minimum_one_unknown_per_two_known",
  "known_ids": ["p001", "p004", "p005", "p006", "p007", "p010", "p011", "p012", "p013", "p014", "p019"],
  "unknown_ids": ["p002", "p003", "p008", "p009", "p015", "p016", "p017", "p018"],
  "known_sessions": ["session_01", "session_02", "session_03", "session_04"],
  "unknown_sessions": ["session_01", "session_02"],
  "capture_protocol": {
    "warmup_results": 5,
    "samples_per_condition": 20,
    "minimum_sample_interval_ms": 250
  },
  "representation": "all_xyz_pose_v2"
}
```

Prepare a new raw root containing exactly the explicitly selected identities and
sessions. Copy each entire session directory, including its manifest. Keep the
original raw roots intact. The example assumes prior selected captures already
exist under the Windows data root.

```powershell
$Population = Join-Path $Run 'population_01.json'
$ExistingRaw = '..\face-recognition-data\data\raw'
$SelectedRaw = Join-Path $Run 'raw_population_01'
$Spec = Get-Content -Raw -LiteralPath $Population | ConvertFrom-Json
New-Item -ItemType Directory -Path $SelectedRaw -ErrorAction Stop | Out-Null
foreach ($Id in ($Spec.known_ids + $Spec.unknown_ids)) {
    $Source = if ($Id -eq 'p019') { $Incoming } else { $ExistingRaw }
    $Sessions = if ($Id -in $Spec.known_ids) { $Spec.known_sessions } else { $Spec.unknown_sessions }
    $Destination = Join-Path $SelectedRaw $Id
    New-Item -ItemType Directory -Path $Destination -ErrorAction Stop | Out-Null
    foreach ($Session in $Sessions) {
        Copy-Item -LiteralPath (Join-Path $Source "$Id\$Session") -Destination $Destination -Recurse -ErrorAction Stop
    }
}

$Processed = Join-Path $Run 'pose_processed_01'
$Bundle = Join-Path $Run 'models\dynamic_model_001'
python -m face_recognition.presentation.cli.process_dataset --raw-root "$SelectedRaw" --output-root "$Processed" --population-config "$Population"
python -m face_recognition.presentation.cli.train_dynamic_model --processed-root "$Processed" --population-config "$Population" --validate-only
python -m face_recognition.presentation.cli.train_dynamic_model --processed-root "$Processed" --population-config "$Population" --output-dir "$Bundle"
python -m face_recognition.presentation.cli.train_dynamic_model --verify-bundle "$Bundle"
```

Check each command's exit status before continuing. Dynamic processing requires
`--population-config` so it can verify complete raw captures and bind their digests
to the processed feature files. Regenerate the entire selected population into
a fresh directory. An old processed root or a partial update does not carry
the required provenance. Preflight reads and validates all payloads but fits no
estimators. Training uses the complete selected data, so RAM and time grow with
population size. Run fitting on the development machine.

The processing manifest carries a versioned `dynamic_capture` entry with the
canonical population SHA-256, the aggregate SHA-256 over relative raw filenames
and their checksums, and the corresponding processed-file checksum aggregate.
The builder verifies the processed catalog, manifests and feature bytes before
fitting and rechecks inputs before publication. Keep source directories stable
through processing and training.

The immutable bundle has `model.joblib`, `config.json` and `manifest.json`.
Schema 1 with model ID `open_set_model_dynamic` is distinct from the historical
V3 bundle. Its configuration records the sorted roles and capture policy, source
digests, selected algorithm and implementation fingerprints. Its manifest binds
the files, exact class order, allocation, counts, threshold, learned-state hashes,
runtime versions and calibration-only metrics. Paths to raw data, landmark rows,
individual features and human-readable names are not included in the manifest.
Only load trusted local bundles. Checksums detect inconsistency and corruption,
but do not establish who produced a joblib artifact.

## Deploy and roll back on the Pi

Create a fresh parent directory on the Pi, then transfer the whole bundle from
Windows. Keep the existing model and its original launcher settings available.

```bash
# On the Pi
mkdir -p "$HOME/Projects/face-recognition-data/retraining/models"
```

```powershell
# On Windows, after confirming the chosen Pi bundle directory does not exist
scp -r "$Bundle" "${Pi}:~/Projects/face-recognition-data/retraining/models/"
```

On the Pi, verify the copied bundle in the kiosk environment before starting it.
The loader checks Python major/minor, exact NumPy, SciPy and scikit-learn versions,
source compatibility, class order and learned-state integrity. A mismatch requires
a compatible environment or a newly constructed compatible bundle.

```bash
cd ~/Projects/face-recognition
source .venv/bin/activate
export PYTHONPATH=src
MODEL="$HOME/Projects/face-recognition-data/retraining/models/dynamic_model_001"
DB="$HOME/.local/share/face-recognition/live-pilot-attendance.sqlite3"
python -m face_recognition.presentation.cli.train_dynamic_model --verify-bundle "$MODEL"

# From the physical console, with an explicitly selected authorized database.
QT_QPA_PLATFORM=linuxfb:fb=/dev/fb0 QT_QPA_FB_DRM=0 \
PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 \
python -m face_recognition.presentation.gui \
  --camera-backend picamera2 --model-dir "$MODEL" --attendance-db "$DB" \
  --orientation rotate180 --preview-fps 15
```

Open an appropriate test session through the administrator panel before testing
recognition. The command omits `--enable-attendance-recording`, so recognition
results do not write attendance. Verify accepted KNOWN and rejected UNKNOWN
behavior before updating the device-local launcher's model path. For a controlled
authorized recording check, add `--enable-attendance-recording` and verify that
a selected KNOWN ID is already registered in SQLite. UNKNOWN identities require
no SQLite registration. Use the existing session and consent rules.

To roll back, exit normally and restore the previous model path in the local
launcher or command. Preserve that complete bundle and verify it in the same
runtime. No bundle merging, attendance database migration, or training on the Pi
is required.
