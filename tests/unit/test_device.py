"""Synthetic /proc and sensors; operating-system power commands are always mocked."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from face_recognition.infrastructure.device import DeviceDiagnostics, SystemPowerController


class DeviceTests(unittest.TestCase):
    def test_cpu_deltas_memavailable_temperature_and_storage(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'stat').write_text('cpu 100 0 100 800 0 0 0 0 0 0\n')
            (root/'meminfo').write_text('MemTotal: 1024 kB\nMemAvailable: 512 kB\nMemFree: 100 kB\nHugePages_Total: 0\n')
            (root/'temp').write_text('42500')
            probe=DeviceDiagnostics(root,proc_root=root,thermal_path=root/'temp',platform='linux')
            first=probe.sample(); self.assertIsNone(first['cpu'])
            self.assertEqual(first['ram_used'],512*1024); self.assertEqual(first['temperature'],42.5)
            self.assertGreater(first['storage'].total,0)
            (root/'stat').write_text('cpu 150 0 150 900 0 0 0 0 0 0\n')
            self.assertEqual(probe.sample()['cpu'],50.)
            (root/'temp').write_text('nan')
            (root/'stat').write_text('invalid')
            (root/'meminfo').write_text('MemTotal: invalid kB')
            result=probe.sample()
            self.assertIsNone(result['temperature']); self.assertIsNone(result['cpu']); self.assertIsNone(result['ram_used'])

    def test_unsupported_and_missing_sensors_are_unavailable(self):
        with TemporaryDirectory() as temp:
            result=DeviceDiagnostics(temp,platform='win32').sample()
            self.assertIsNone(result['cpu']); self.assertIsNone(result['temperature']); self.assertIsNone(result['ram_total'])
            result=DeviceDiagnostics(temp,proc_root=Path(temp)/'missing',thermal_path=Path(temp)/'missing',platform='linux').sample()
            self.assertIsNone(result['cpu']); self.assertIsNone(result['temperature'])

    def test_power_disabled_unsupported_arguments_and_command_failures(self):
        with patch('face_recognition.infrastructure.device.subprocess.run') as run:
            for action in ('reboot','poweroff'):
                with self.assertRaises(PermissionError): SystemPowerController().execute(action)
            run.assert_not_called()
            with self.assertRaises(ValueError): SystemPowerController(enabled=True,platform='win32')
            power=SystemPowerController(enabled=True,platform='linux')
            for action in ('reboot','poweroff'):
                power.execute(action)
                self.assertEqual(run.call_args.args[0],['systemctl',action])
                self.assertNotIn('shell',run.call_args.kwargs)
            with self.assertRaises(ValueError): power.execute('other')
            run.side_effect=PermissionError('Denied')
            with self.assertRaises(PermissionError): power.execute('reboot')
