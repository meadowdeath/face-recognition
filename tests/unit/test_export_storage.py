"""Synthetic UDisks2 properties and failure-safe file publication; no physical USB."""
from dataclasses import replace
import errno
import importlib.util
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from face_recognition.domain.models.export import ExportError
AVAILABLE = importlib.util.find_spec('PyQt5') is not None
if AVAILABLE:
    from PyQt5.QtCore import QByteArray
    from PyQt5.QtDBus import QDBusVariant
    from face_recognition.infrastructure.udisks import (UDisksProvider, DRIVE, BLOCK, FS, decoded,
        destinations, mount_path, mounts, options, physical_identity)
from face_recognition.infrastructure.persistence.export_files import publish, FAT32_LIMIT
from tests.export_fakes import ExportFixture


def objects():
    return {'/drive': {DRIVE: dict(ConnectionBus='usb', Removable=True, MediaRemovable=False,
             MediaAvailable=True, CanPowerOff=True, Id='synthetic', Serial='fictional', TimeDetected=1,
             Vendor='Ficticio', Model='USB', SiblingId='group')},
            '/block': {BLOCK: dict(Drive='/drive', Device=b'/dev/synthetic1\0', DeviceNumber=os.makedev(8, 1) if hasattr(os, 'makedev') else 2049,
                      HintSystem=False, HintIgnore=False, IdUsage='filesystem', IdType='vfat', IdUUID='fictional',
                      IdLabel='Prueba con espacios Á', Size=8*2**30, ReadOnly=False), FS: dict(MountPoints=[])}}


@unittest.skipUnless(AVAILABLE, 'Optional Qt dependency unavailable')
class UDisksTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt5.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        # Synthetic Linux dev_t values even when these pure adapter tests run on Windows.
        if not hasattr(os, 'makedev'):
            for name, implementation in (('makedev', lambda major, minor: major*256+minor),
                                         ('major', lambda number: number//256), ('minor', lambda number: number%256)):
                self.enterContext(patch.object(os, name, implementation, create=True))

    def test_partitioned_and_unpartitioned_usb_media_removable_false(self):
        data = objects()
        found = destinations(data, [], [], 1)
        self.assertEqual(len(found), 1); self.assertIn('espacios Á', found[0].label)
        data['/block'][BLOCK]['Device'] = b'/dev/synthetic\0'
        self.assertEqual(len(destinations(data, [], [], 1)), 1)
        self.assertNotEqual(found[0].identity, destinations(data, [], [], 2)[0].identity)

    def test_mount_decoding_spaces_unicode_variants_and_kernel_identity(self):
        self.assertEqual(mount_path(QByteArray('/media/Prueba Á\0'.encode())), '/media/Prueba Á')
        self.assertEqual(decoded({'v': QDBusVariant(QByteArray(b'a\0'))}), {'v': b'a\0'})
        rows = mounts('21 1 8:1 / /media/Prueba\\040A rw,nosuid - vfat /dev/synthetic1 rw')
        self.assertEqual(rows[0]['path'], '/media/Prueba A')
        data = objects(); data['/block'][FS]['MountPoints'] = [b'/media/Prueba A\0']
        found = destinations(data, rows, [], 1)
        self.assertEqual(found[0].mount_id, '21')
        self.assertFalse(destinations(data, [dict(rows[0], number='8:2')], [], 1))
        self.assertFalse(destinations(data, rows+[rows[0]], [], 1))
        for value in (b'relative\0', b'/bad\0embedded\0', '/string'):
            with self.assertRaises(ValueError): mount_path(value)

    def test_system_internal_optical_ignored_readonly_and_multiple_devices(self):
        for section, key, value in ((DRIVE, 'ConnectionBus', 'mmc'), (DRIVE, 'Optical', True),
                                   (DRIVE, 'MediaAvailable', False), (BLOCK, 'HintSystem', True),
                                   (BLOCK, 'HintIgnore', True), (BLOCK, 'IdUsage', 'swap'), (BLOCK, 'IdType', 'unknown')):
            data = objects(); data['/drive' if section == DRIVE else '/block'][section][key] = value
            self.assertFalse(destinations(data, [], [], 1))
        data = objects(); data['/block'][BLOCK]['ReadOnly'] = True
        self.assertTrue(destinations(data, [], [], 1)[0].read_only)
        data['/other'] = {**data['/block'], BLOCK: {**data['/block'][BLOCK], 'DeviceNumber': os.makedev(8, 2)}}
        self.assertEqual(len(destinations(data, [], [], 1)), 2)
        self.assertFalse(destinations(data, mounts('1 0 8:1 / / rw - vfat /dev/synthetic1 rw'), [], 1))

    def test_mount_calls_noninteractive_options_and_eject_sibling_safety(self):
        provider = UDisksProvider([], platform='win32')
        d = destinations(objects(), [], [], 0)[0]
        with patch.object(provider, 'validate', return_value=d), patch.object(provider, 'call') as call, \
             patch.object(provider, 'discover', return_value=(replace(d, mount='/synthetic'),)):
            provider.mount(d)
            self.assertEqual(call.call_args.args[:3], ('/block', FS, 'Mount'))
            self.assertIsInstance(call.call_args.args[3][0], type(options()))
        d = replace(d, mount='/synthetic', drive_identity=physical_identity(objects()['/drive'][DRIVE], 0))
        data = objects(); data['/other'] = {BLOCK: dict(Drive='/drive'), FS: dict(MountPoints=[b'/other\0'])}
        with patch.object(provider, 'validate', return_value=d), patch.object(provider, 'objects', return_value=data), patch.object(provider, 'call') as call:
            self.assertIn('Otra partición', provider.eject(d))
            self.assertEqual(call.call_count, 1)
        with patch.object(provider, 'validate', return_value=d), patch.object(provider, 'objects', return_value=objects()), \
             patch.object(provider, 'call', side_effect=[[], ExportError('authorization', 'Sin autorización')]):
            self.assertIn('no se confirmó el apagado', provider.eject(d))
        provider.owner_changed('synthetic', 'old', '')
        self.assertEqual(provider._generation, 1)

    def test_bounded_qtdbus_error_replies_denial_timeout_busy_and_service_loss(self):
        from PyQt5.QtDBus import QDBusPendingCall, QDBusMessage
        from unittest.mock import Mock
        bus = Mock(); bus.isConnected.return_value = True
        provider = UDisksProvider([], platform='win32'); provider.bus = bus; provider.platform = 'linux'
        for name, code in (('NotAuthorized', 'authorization'), ('NoReply', 'timeout'),
                           ('DeviceBusy', 'busy'), ('NotSupported', 'filesystem'), ('ServiceUnknown', 'service')):
            reply = QDBusMessage.createError('org.freedesktop.UDisks2.Error.'+name, 'synthetic secret path must not be shown')
            bus.asyncCall.return_value = QDBusPendingCall.fromCompletedCall(reply)
            with self.assertRaises(ExportError) as failure: provider.call('/synthetic', FS, 'Mount', [options()])
            self.assertEqual(failure.exception.code, code)
            self.assertNotIn('synthetic secret', str(failure.exception))
            self.assertEqual(bus.asyncCall.call_args.args[1], 10000)
        bus.isConnected.return_value = False
        with self.assertRaises(ExportError): provider.objects()

    def test_eject_rejects_replacement_drive_and_mounted_sibling(self):
        provider = UDisksProvider([], platform='win32')
        data = objects(); d = destinations(data, [], [], 0)[0]
        data['/sibling'] = {DRIVE: dict(SiblingId='group')}
        data['/siblingblock'] = {BLOCK: dict(Drive='/sibling'), FS: dict(MountPoints=[b'/other\0'])}
        with patch.object(provider, 'validate', return_value=d), patch.object(provider, 'objects', return_value=data), patch.object(provider, 'call') as call:
            self.assertIn('Otra partición', provider.eject(d)); call.assert_not_called()
        data['/drive'][DRIVE]['TimeDetected'] = 2
        with patch.object(provider, 'validate', return_value=d), patch.object(provider, 'objects', return_value=data), patch.object(provider, 'call') as call:
            with self.assertRaises(ExportError): provider.eject(d)
            call.assert_not_called()


class PublicationTests(ExportFixture):
    def request(self): return self.service.prepare('report', 'CSV', self.provider.current).request

    def test_never_overwrites_and_cleans_collision_temporary(self):
        request = self.request(); first = self.execute(request)
        path = self.usb/'FaceID/registros'/first.filename; before = path.read_bytes()
        with self.assertRaises((ExportError, FileExistsError)): self.execute(request)
        self.assertEqual(path.read_bytes(), before)
        self.assertFalse(list(path.parent.glob('*.tmp')))

    def test_path_traversal_and_absolute_names_rejected(self):
        request = self.request()
        for name in ('../other.csv', '/outside.csv', 'a\\b.csv', 'a/../b.csv'):
            with self.assertRaises(ValueError): replace(request, filename=name)

    def test_symlink_directories_rejected_without_writing_elsewhere(self):
        with TemporaryDirectory() as other:
            try: (self.usb/'FaceID').symlink_to(other, target_is_directory=True)
            except OSError: self.skipTest('Symlink creation unavailable')
            with self.assertRaises((OSError, ExportError)): self.execute(self.request())
            self.assertFalse(list(Path(other).iterdir()))

    def test_readonly_space_fat32_limit_and_cancellation(self):
        request = self.request(); staged = self.root/'staged'; staged.write_bytes(b'abc')
        self.provider.current = replace(request.destination, read_only=True)
        with self.assertRaises(ExportError):
            publish(staged, request, self.provider, self.cancel, lambda *a: None)
        fat = replace(request, destination=replace(request.destination, filesystem='vfat'))
        self.provider.current = fat.destination
        from types import SimpleNamespace
        original = Path.stat
        def size(path, *args, **kwargs):
            return SimpleNamespace(st_size=FAT32_LIMIT+1) if path == staged else original(path, *args, **kwargs)
        with patch.object(Path, 'stat', size), self.assertRaises(ExportError): publish(staged, fat, self.provider, self.cancel, lambda *a: None)
        self.provider.current = request.destination
        self.cancel.set()
        with self.assertRaises(ExportError): publish(staged, request, self.provider, self.cancel, lambda *a: None)
        self.assertFalse(list(self.usb.rglob('*.tmp')))

    def test_disconnect_publication_failure_and_sync_failures_never_claim_success(self):
        staged = self.root/'staged'; staged.write_bytes(b'abc'); request = self.request()
        with patch('face_recognition.infrastructure.persistence.export_files.os.fsync', side_effect=OSError(errno.EIO, 'synthetic')):
            with self.assertRaises(ExportError): publish(staged, request, self.provider, self.cancel, lambda *a: None)
        self.assertFalse(list(self.usb.rglob('*.tmp')))
        def disconnect(*args): self.provider.connected = False
        with self.assertRaises(ExportError): publish(staged, request, self.provider, self.cancel, disconnect)
        self.assertFalse(list(self.usb.rglob('*.csv')))
        self.provider.connected = True
        with patch('face_recognition.infrastructure.persistence.export_files.rename_exclusive', side_effect=ExportError('publication', 'No compatible')):
            with self.assertRaises(ExportError): publish(staged, request, self.provider, self.cancel, lambda *a: None)
        self.assertFalse(list(self.usb.rglob('*.tmp')))

    def test_final_sync_failure_reports_published_filename_and_uncertain_durability(self):
        staged = self.root/'staged'; staged.write_bytes(b'abc'); request = self.request()
        original = self.provider.validate; calls = 0
        def validate(destination):
            nonlocal calls
            calls += 1
            if calls == 6: raise ExportError('replacement', 'synthetic')
            return original(destination)
        with patch.object(self.provider, 'validate', side_effect=validate):
            with self.assertRaisesRegex(ExportError, request.filename): publish(staged, request, self.provider, self.cancel, lambda *a: None)
        self.assertTrue((self.usb/'FaceID/registros'/request.filename).exists())

    def test_insufficient_space_and_mount_replacement_leave_no_final_file(self):
        from types import SimpleNamespace
        staged = self.root/'staged'; staged.write_bytes(b'abc'); request = self.request()
        target = ('face_recognition.infrastructure.persistence.export_files.os.fstatvfs' if os.name != 'nt'
                  else 'face_recognition.infrastructure.persistence.export_files.shutil.disk_usage')
        free = SimpleNamespace(f_bavail=0, f_frsize=4096, free=0)
        with patch(target, return_value=free), self.assertRaises(ExportError):
            publish(staged, request, self.provider, self.cancel, lambda *a: None)
        original = self.provider.validate; calls = 0
        def replaced(destination):
            nonlocal calls
            calls += 1
            if calls == 5: self.provider.current = replace(self.provider.current, identity='replacement')
            return original(destination)
        with patch.object(self.provider, 'validate', side_effect=replaced), self.assertRaises(ExportError):
            publish(staged, request, self.provider, self.cancel, lambda *a: None)
        self.assertFalse(list(self.usb.rglob('*.csv'))); self.assertFalse(list(self.usb.rglob('*.tmp')))

    def test_cancel_during_final_destination_revalidation_prevents_publication(self):
        staged = self.root/'staged'; staged.write_bytes(b'abc'); request = self.request()
        original = self.provider.validate; calls = 0
        def validated(destination):
            nonlocal calls
            calls += 1
            if calls == 5: self.cancel.set()
            return original(destination)
        with patch.object(self.provider, 'validate', side_effect=validated), self.assertRaises(ExportError):
            publish(staged, request, self.provider, self.cancel, lambda *a: None)
        self.assertFalse(list(self.usb.rglob('*.csv'))); self.assertFalse(list(self.usb.rglob('*.tmp')))
