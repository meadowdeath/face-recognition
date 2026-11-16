# Raspberry Pi kiosk startup stability

LinuxFB is the recommended Qt platform backend for the current Raspberry Pi
kiosk prototype and academic demonstration. EGLFS remains a historical or
experimental alternative: an intermittent native startup crash was observed.
This note records operator-supplied evidence and validation, not hardware tests
performed by Codex. It does not begin Phase 8 or Phase 9 deployment work.

## Incident environment and observations

- Raspberry Pi 3 Model B Plus, 1 GB RAM; Camera Module 3 (IMX708).
- HDMI display, 800 × 480; Debian 12 AArch64.
- Linux `6.12.115+rpt-rpi-v8`; Python 3.11.
- System-provided PyQt5 / Qt 5.15; NumPy 1.26.4.
- Qt EGLFS with KMS integration, Mesa and GBM graphics stack.

Startup intermittently terminated with a segmentation fault and exit status 139.
The physical display remained frozen after the Python process terminated, while
SSH remained accessible: the operating system was still responsive. Some starts
succeeded after rebooting. Subsequent application launches appeared more likely
to fail, but that pattern was not conclusively established.

The failure preceded camera activation and completion of recognition-model
loading. Python faulthandler showed the main thread in `window.showFullScreen()`
in `src/face_recognition/presentation/gui/__main__.py`, while the recognition
worker was importing dependencies. Five independent NumPy import checks passed;
preloading NumPy before Qt initialization did not prevent the crash.

GDB identified this native sequence, with the faulting function in the
system-provided `libQt5Gui.so.5`:

```text
QWidget::showFullScreen()
QWidget::setWindowState()
QWidget::create()
QWidgetPrivate::create()
QWindowPrivate::create()
QEglFSIntegration::createPlatformWindow()
QEglFSWindow::create()
QOpenGLContext::create()
QEglFSIntegration::createPlatformOpenGLContext()
QOpenGLContext::handle()
SIGSEGV
```

The recognition worker was still importing Python modules when the main thread
received SIGSEGV. Post-failure measurements were 764 MiB available RAM, 0 B swap
usage, 36.5 °C and throttling status `0x0`. These measurements were collected
**after** the failure and do not conclusively exclude transient resource pressure.

The observed failure occurred during EGLFS OpenGL context creation. The underlying
Qt/driver defect, race or precise resource-lifecycle problem has **not** been
identified. Concurrent initialization is a possible contributor, not a confirmed
root cause. The evidence does not establish NumPy, MediaPipe, Picamera2, SQLite or
the V3 classifier as the cause.

## Validated workaround and its limits

The operator selected:

```bash
export QT_QPA_PLATFORM=linuxfb:fb=/dev/fb0
export QT_QPA_FB_DRM=0
```

LinuxFB supports software-rendered Qt Widgets without the EGLFS OpenGL context
creation path. The interface uses Widgets, QImage and QPixmap and requires no
explicit OpenGL rendering surface. The existing entry point calls
`showFullScreen()` only for EGLFS; LinuxFB uses the existing `show()` path with
the fixed 800 × 480 window. No startup-worker refactoring is part of this workaround.

This environment selection changes no recognition architecture, frozen V3 model,
camera capture, MediaPipe inference, feature extraction/normalization, thresholds,
temporal decisions, SQLite persistence, administrative authentication, USB exports,
backups or project dependencies. Windows native Qt and automated offscreen testing
remain unchanged.

Operator-confirmed validation:

- The LinuxFB diagnostic application started successfully.
- Live facial recognition worked with LinuxFB.
- The initial experimental LinuxFB launcher deliberately disabled attendance
  recording. The operational launcher was subsequently updated to LinuxFB while
  retaining `--enable-attendance-recording`.
- At least five consecutive application launches succeeded without rebooting
  between them.
- At least one successful attendance registration was confirmed.

Pending work includes extended continuous/long-duration operation, comparative
CPU/memory/FPS/latency measurements, comprehensive feature regression under
LinuxFB, and definitive investigation of the EGLFS defect. Earlier USB and
administration validation remains historical evidence; it is not newly performed
LinuxFB-specific validation. Repeated successful launches validate a workaround,
not a permanent repair or a guarantee of long-term stability.

## Controlled pilot execution

Launch from the Raspberry Pi's **physical local console** in the currently
validated configuration. The trusted model bundle and required MediaPipe asset
must already exist. The external attendance database must already be initialized
at schema v4, with administrator credentials configured and authorized participant
IDs explicitly registered. Recording requires an active attendance session and
Attendance lifecycle state. Startup remains in Standby.

The following reproduces the supplied operational launcher settings. Paths are
deployment examples only; this documentation task did not inspect these files:

```bash
cd ~/Projects/face-recognition
source .venv/bin/activate

export QT_QPA_PLATFORM=linuxfb:fb=/dev/fb0
export QT_QPA_FB_DRM=0
export PYTHONPATH=src
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1

MODEL="$HOME/Projects/face-recognition-data/v3/models/open_set_model_v3"
DB="$HOME/.local/share/face-recognition/live-pilot-attendance.sqlite3"

python -m face_recognition.presentation.gui \
  --camera-backend picamera2 \
  --model-dir "$MODEL" \
  --attendance-db "$DB" \
  --enable-attendance-recording \
  --orientation rotate180 \
  --preview-fps 15
```

This is a **controlled pilot database**, not production deployment. Live attendance
tests or demonstrations require appropriately authorized participants, reviewed
purposes and consent. Do not use operational data for automated tests or publish
participant information, model artifacts or database contents.

Add `--debug` for diagnostic performance output. Omit
`--enable-attendance-recording` for recognition testing without attendance writes;
with a database, an open session still gates camera activation. This flag omission
does not make other administrative actions read-only. System power requests remain
disabled unless explicitly enabled; the command above does not enable them.
No Qt plugin-path override or other unvalidated Qt environment setting is required.

Exit through authenticated administrator navigation and its confirmation. There
is no global Escape-to-close shortcut or unauthenticated exit fallback.

## Existing device-local launchers

The scripts below are outside the repository. Their responsibilities are reported
by the operator; their contents were not inspected here, and no replacement scripts
are supplied or tracked by this change.

`~/.local/bin/start-face-id` locates the project, activates the existing virtual
environment, selects LinuxFB and configures Python. It checks the required V3
model files and the external SQLite database, expected schema version and database
integrity, administrator-credential existence and registered-participant existence.
It launches the integrated application with recording enabled and accepts optional
`--debug`. These checks do not themselves establish participant authorization or
consent, and do not replace the application's artifact/authentication validation.

`~/.local/bin/diagnose-face-id` invokes `start-face-id --debug`, enables Python
faulthandler and unbuffered output, stores timestamped diagnostic logs under
`~/face-id-logs/`, preserves application exit status and uses restrictive permissions
for newly created logs. Exact implementation is device-local configuration.

```bash
~/.local/bin/start-face-id
~/.local/bin/start-face-id --debug
~/.local/bin/diagnose-face-id
```

## Troubleshooting without changing system configuration

Use the diagnostic launcher or the faulthandler/unbuffered environment above.
Preserve the first failure and its exit status. Logs and native backtraces may
contain sensitive application information or paths: keep them private, outside
Git, and review/redact before sharing. No private device logs were accessed here.

Read-only checks from an authorized console or SSH session include:

```bash
pgrep -af '[p]ython.*face_recognition.presentation.gui'
journalctl --list-boots
journalctl -b --since "10 minutes ago"
journalctl -k -b --since "10 minutes ago"
free -h
vcgencmd measure_temp
vcgencmd get_throttled
```

A frozen display does not prove that Python is still running; check the process
and SSH responsiveness separately. Journal visibility depends on existing
permissions. Previous-boot journal evidence is available only if persistent
systemd journaling was already configured and retained it; no journald/systemd
configuration change is prescribed. A manually launched application's terminal
output is not guaranteed to appear in the journal: retain the launcher's timestamped
logs separately. Collect measurements during an incident when possible; measurements
afterwards cannot establish conditions at the instant of failure.

GDB is an optional native-crash investigation tool when already available. Use an
approved diagnostic invocation (preferably without attendance recording), then
`run` and `thread apply all bt` after a fault. Preserve the native stack and thread
states privately; Python faulthandler alone does not identify the underlying native
defect. Do not infer a cause from another thread merely importing a module.

Do not use automatic restarts as a substitute for diagnosis. This note adds no
systemd deployment, automatic startup, touchscreen setup, OS configuration,
dependency upgrades, database migration, model retraining or new application feature.
