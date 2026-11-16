"""Small local diagnostics and opt-in Linux power operations; no recognition imports."""
from pathlib import Path
import shutil
import subprocess
import sys
from math import isfinite


class DeviceDiagnostics:
    def __init__(self, storage_path, *, proc_root=Path('/proc'), thermal_path=Path('/sys/class/thermal/thermal_zone0/temp'), platform=None):
        self.storage_path = Path(storage_path)
        self.proc_root, self.thermal_path = Path(proc_root), Path(thermal_path)
        self.platform = platform or sys.platform
        self._cpu = None

    def sample(self):
        result = dict(cpu=None, temperature=None, ram_total=None, ram_used=None, ram_available=None, storage=None)
        if self.platform.startswith('linux'):
            try:
                row = (self.proc_root / 'stat').read_text().splitlines()[0].split()
                counters = [int(value) for value in row[1:9]]
                if row[0] != 'cpu' or len(counters) < 4 or min(counters) < 0:
                    raise ValueError('Invalid CPU counters')
                total, idle = sum(counters), counters[3] + (counters[4] if len(counters) > 4 else 0)
                if self._cpu is not None:
                    elapsed, idle_elapsed = total-self._cpu[0], idle-self._cpu[1]
                    if elapsed > 0 and 0 <= idle_elapsed <= elapsed:
                        result['cpu'] = 100 * (elapsed-idle_elapsed) / elapsed
                self._cpu = total, idle
            except (OSError, ValueError, IndexError):
                self._cpu = None
            try:
                memory = {}
                for line in (self.proc_root / 'meminfo').read_text().splitlines():
                    if not line.startswith(('MemTotal:', 'MemAvailable:')):
                        continue
                    key, value, unit = line.split()
                    if unit == 'kB': memory[key.rstrip(':')] = int(value) * 1024
                total, available = memory['MemTotal'], memory['MemAvailable']
                if not 0 <= available <= total or total <= 0: raise ValueError('Invalid memory counters')
                result.update(ram_total=total, ram_available=available, ram_used=total-available)
            except (OSError, ValueError, KeyError):
                pass
            try:
                temperature = float(self.thermal_path.read_text()) / 1000
                if isfinite(temperature) and -20 <= temperature <= 150:
                    result['temperature'] = temperature
            except (OSError, ValueError):
                pass
        try:
            result['storage'] = shutil.disk_usage(self.storage_path)
        except OSError:
            pass
        return result


class SystemPowerController:
    def __init__(self, *, enabled=False, platform=None):
        self.enabled = enabled
        self.platform = platform or sys.platform
        if enabled and not self.platform.startswith('linux'):
            raise ValueError('System power actions require Linux')

    def execute(self, action):
        if not self.enabled or not self.platform.startswith('linux'):
            raise PermissionError('System power actions are disabled')
        if action not in ('reboot', 'poweroff'):
            raise ValueError('Unsupported power action')
        # Called only after the Qt event loop and recognition worker have stopped.
        subprocess.run(['systemctl', action], check=True, capture_output=True, timeout=15)
