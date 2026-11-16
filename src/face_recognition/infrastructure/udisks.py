"""UDisks2 via QtDBus only. All bounded calls are made by the export worker.

PyQt5 recursively demarshals QDBusArgument maps/arrays/variants in message
arguments (qpydbus_from_qvariant_convertor); no dbus-python dependency is used.
"""
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import sys
from threading import RLock

from PyQt5.QtCore import QObject, QEventLoop, QTimer, QMetaType, QByteArray, QVariant, pyqtSignal, pyqtSlot
from PyQt5.QtDBus import (QDBusArgument, QDBusConnection, QDBusMessage,
                         QDBusPendingCallWatcher, QDBusPendingReply, QDBusServiceWatcher, QDBusVariant)

from face_recognition.domain.models.export import Destination, ExportError

SERVICE = 'org.freedesktop.UDisks2'
ROOT = '/org/freedesktop/UDisks2'
DRIVE, BLOCK, FS = (SERVICE+'.'+name for name in ('Drive', 'Block', 'Filesystem'))
SUPPORTED = {'vfat', 'fat', 'fat32', 'exfat', 'ntfs', 'ext2', 'ext3', 'ext4'}


def physical_identity(drive, generation):
    return sha256(json.dumps((generation, drive.get('Id'), drive.get('Serial'),
                              drive.get('TimeDetected'))).encode()).hexdigest()


def options():
    """Explicit a{sv}, rather than QVariantMap with incorrectly typed bool values."""
    arg = QDBusArgument()
    arg.beginMap(QMetaType.QString, QVariant(QDBusVariant(True)).userType())
    arg.beginMapEntry(); arg.add('auth.no_user_interaction'); arg.add(QDBusVariant(True))
    arg.endMapEntry(); arg.endMap()
    return arg


def decoded(value):
    if isinstance(value, QDBusVariant): return decoded(value.variant())
    if isinstance(value, dict): return {str(k): decoded(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [decoded(v) for v in value]
    if isinstance(value, QByteArray): return bytes(value)
    return value


def mount_path(value):
    if isinstance(value, (list, tuple)) and all(type(b) is int and 0 <= b <= 255 for b in value):
        value = bytes(value)
    if not isinstance(value, (bytes, bytearray, QByteArray)):
        raise ValueError('Invalid mount point byte array')
    raw = bytes(value).rstrip(b'\0')
    if b'\0' in raw: raise ValueError('Embedded NUL')
    path = os.fsdecode(raw)
    if not path.startswith('/') or '..' in Path(path).parts:
        raise ValueError('Invalid mount point')
    return path


def mounts(text):
    """Parse kernel mount identity; decode mountinfo's octal path escapes."""
    result = []
    for line in text.splitlines():
        fields = line.split(); split = fields.index('-')
        if split < 6 or len(fields) < split+4: raise ValueError('Invalid mount metadata')
        unescape = lambda s: re.sub(r'\\([0-7]{3})', lambda m: chr(int(m[1], 8)), s)
        result.append(dict(id=fields[0], number=fields[2], root=unescape(fields[3]),
                           path=unescape(fields[4]), options=fields[5].split(','),
                           filesystem=fields[split+1], source=unescape(fields[split+2])))
    return result


def destinations(objects, mount_rows, protected, generation):
    """Fail closed for unverifiable drives, layered mappings and protected mounts."""
    if not isinstance(objects, dict): raise ValueError('Invalid managed objects')
    excluded = set()
    protected_numbers = {row['number'] for row in mount_rows if row['path'] in ('/', '/boot', '/boot/firmware')}
    for path in protected:
        device = os.stat(path).st_dev
        protected_numbers.add(f'{os.major(device)}:{os.minor(device)}')
    for interfaces in objects.values():
        block = interfaces.get(BLOCK, {})
        number = block.get('DeviceNumber')
        if isinstance(number, int) and f'{os.major(number)}:{os.minor(number)}' in protected_numbers:
            excluded.add(block.get('Drive'))
        if any(m['number'] in protected_numbers and m['path'] == mount_path(point)
               for point in interfaces.get(FS, {}).get('MountPoints', []) for m in mount_rows):
            excluded.add(block.get('Drive'))
    result = []
    for path, interfaces in objects.items():
        if FS not in interfaces or BLOCK not in interfaces: continue
        b, f = interfaces[BLOCK], interfaces[FS]
        drive_path = b.get('Drive'); d = objects.get(drive_path, {}).get(DRIVE, {})
        if (drive_path in excluded or d.get('ConnectionBus') != 'usb' or not d.get('MediaAvailable')
                or d.get('Optical', False) or b.get('HintSystem', True) or b.get('HintIgnore', True)
                or b.get('IdUsage') != 'filesystem' or b.get('IdType') not in SUPPORTED
                or not isinstance(b.get('DeviceNumber'), int) or b['DeviceNumber'] <= 0
                or not (d.get('Id') or d.get('Serial')) or not d.get('TimeDetected')):
            continue
        device = mount_path(b['Device'])
        if not device.startswith('/dev/') or device.startswith(('/dev/loop', '/dev/sr', '/dev/mmc')): continue
        points = [mount_path(p) for p in f.get('MountPoints', [])]
        if len(points) > 1: continue
        point, match = (points[0] if points else ''), None
        if point:
            matches = [m for m in mount_rows if m['path'] == point]
            number = b['DeviceNumber']
            if len(matches) != 1: continue
            kernel = matches[0]
            fuse = (kernel['filesystem'] == 'fuseblk' and b['IdType'] == 'ntfs' and kernel['source'] == device)
            if ((kernel['number'] != f'{os.major(number)}:{os.minor(number)}' and not fuse)
                    or kernel['root'] != '/' or kernel['filesystem'] not in SUPPORTED | {'fuseblk'}):
                continue
            match = matches[0]
        stamp = (generation, path, drive_path, d.get('Id'), d.get('Serial'), d.get('TimeDetected'),
                 b['DeviceNumber'], device, b.get('IdUUID'), b.get('IdType'), b.get('ReadOnly'), b.get('Size'), point,
                 None if match is None else (match['id'], match['number'], match['root'], match['options']))
        identity = sha256(json.dumps(stamp, sort_keys=True).encode()).hexdigest()
        label = ' '.join(str(v) for v in (d.get('Vendor', ''), d.get('Model', ''), b.get('IdLabel', '')) if v) or 'USB'
        result.append(Destination(path, drive_path, identity, label, b['IdType'], int(b.get('Size', 0)),
                      point, '' if match is None else match['id'], b['DeviceNumber'] if match is None else
                      os.makedev(*map(int, match['number'].split(':'))),
                      bool(b.get('ReadOnly') or match and 'ro' in match['options']),
                      can_power_off=bool(d.get('CanPowerOff')), sibling=d.get('SiblingId', ''),
                      drive_identity=physical_identity(d, generation)))
    return tuple(sorted(result, key=lambda item: item.block))


class UDisksProvider(QObject):
    changed = pyqtSignal()

    def __init__(self, protected, parent=None, *, bus=None, mount_reader=None, platform=None):
        super().__init__(parent)
        self.protected = tuple(Path(p).resolve() for p in protected)
        self.platform = platform or sys.platform
        self.bus = bus
        self.mount_reader = mount_reader or (lambda: Path('/proc/self/mountinfo').read_text())
        self._generation = 0
        self._lock = RLock()
        if self.platform.startswith('linux'):
            self.bus = bus or QDBusConnection.systemBus()
            self.watcher = QDBusServiceWatcher(SERVICE, self.bus, QDBusServiceWatcher.WatchForOwnerChange, self)
            self.watcher.serviceOwnerChanged.connect(self.owner_changed)
            for interface, signal in (('org.freedesktop.DBus.ObjectManager', 'InterfacesAdded'),
                                      ('org.freedesktop.DBus.ObjectManager', 'InterfacesRemoved'),
                                      ('org.freedesktop.DBus.Properties', 'PropertiesChanged')):
                self.bus.connect(SERVICE, '', interface, signal, self.bus_changed)

    @pyqtSlot(str, str, str)
    def owner_changed(self, name, old, new):
        with self._lock: self._generation += 1
        self.changed.emit()

    @pyqtSlot(QDBusMessage)
    def bus_changed(self, message):
        if message.member() in ('InterfacesAdded', 'InterfacesRemoved'):
            with self._lock: self._generation += 1
        self.changed.emit()

    def call(self, path, interface, method, args=()):
        if not self.platform.startswith('linux') or self.bus is None or not self.bus.isConnected():
            raise ExportError('service', 'USB disponible solo con UDisks2 y D-Bus en Linux.')
        message = QDBusMessage.createMethodCall(SERVICE, path, interface, method)
        message.setArguments(list(args))
        pending = self.bus.asyncCall(message, 10000)
        watcher = QDBusPendingCallWatcher(pending)
        loop = QEventLoop(); timer = QTimer(); timer.setSingleShot(True)
        watcher.finished.connect(loop.quit); timer.timeout.connect(loop.quit)
        if not watcher.isFinished(): timer.start(11000); loop.exec_()
        timer.stop()
        if not watcher.isFinished():
            raise ExportError('timeout', 'UDisks2 no respondió a tiempo; actualiza antes de reintentar.')
        reply = QDBusPendingReply(watcher).reply()
        if reply.type() == QDBusMessage.ErrorMessage:
            name = reply.errorName()
            if any(token in name for token in ('NoReply', 'Timeout', 'TimedOut')):
                raise ExportError('timeout', 'UDisks2 no respondió a tiempo; actualiza antes de reintentar.')
            if 'NotAuthorized' in name or 'AccessDenied' in name:
                raise ExportError('authorization', 'La sesión actual no tiene autorización USB. Usa la terminal física local; no se solicitó contraseña Linux.')
            if 'Busy' in name:
                raise ExportError('busy', 'El USB está ocupado. Cierra otros archivos antes de expulsarlo.')
            if 'NotSupported' in name or 'UnknownFilesystem' in name:
                raise ExportError('filesystem', 'El sistema no admite esta operación o sistema de archivos USB.')
            raise ExportError('service', 'UDisks2 no está disponible o rechazó la operación. Actualiza los dispositivos.')
        if reply.type() != QDBusMessage.ReplyMessage:
            raise ExportError('reply', 'Respuesta USB no válida.')
        return decoded(reply.arguments())

    def objects(self):
        reply = self.call(ROOT, 'org.freedesktop.DBus.ObjectManager', 'GetManagedObjects')
        if len(reply) != 1 or not isinstance(reply[0], dict):
            raise ExportError('reply', 'Respuesta de dispositivos USB no válida.')
        return reply[0]

    def discover(self):
        with self._lock: generation = self._generation
        try:
            raw = self.objects()
            found = destinations(raw, mounts(self.mount_reader()), self.protected, generation)
            result = []
            for item in found:
                if item.mount:
                    if Path(item.mount).is_symlink() or os.stat(item.mount).st_dev != item.device: continue
                    b = raw[item.block][BLOCK]
                    if item.device != b['DeviceNumber'] and os.stat(mount_path(b['Device'])).st_rdev != b['DeviceNumber']:
                        continue
                    item = replace(item, free=shutil.disk_usage(item.mount).free)
                result.append(item)
            return tuple(result)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            raise ExportError('metadata', 'No se pudo verificar el montaje USB; actualiza los dispositivos.') from exc

    def validate(self, destination):
        current = next((d for d in self.discover() if d.identity == destination.identity), None)
        if current is None:
            raise ExportError('replacement', 'El USB se desconectó o cambió. Selecciónalo y autoriza de nuevo.')
        return current

    def mount(self, destination):
        current = self.validate(destination)
        if current.mount: return 'El USB ya está montado.'
        self.call(current.block, FS, 'Mount', [options()])
        if not any(d.block == current.block and d.mount for d in self.discover()):
            raise ExportError('mount', 'No se pudo verificar el punto de montaje después de montar.')
        return 'USB montado. Selecciona el destino verificado.'

    def eject(self, destination):
        current = self.validate(destination)
        if current.mount:
            self.call(current.block, FS, 'Unmount', [options()])
        objects = self.objects()
        drive = objects.get(current.drive, {}).get(DRIVE)
        if drive is None: return 'USB desmontado; el dispositivo ya no está disponible.'
        with self._lock: generation = self._generation
        if physical_identity(drive, generation) != current.drive_identity:
            raise ExportError('replacement', 'La unidad cambió durante la expulsión; no se solicitó su apagado.')
        for interfaces in objects.values():
            block = interfaces.get(BLOCK, {})
            related = objects.get(block.get('Drive'), {}).get(DRIVE, {})
            if (block.get('Drive') == current.drive or current.sibling and related.get('SiblingId') == current.sibling):
                if interfaces.get(FS, {}).get('MountPoints'):
                    return 'USB desmontado. Otra partición o unidad relacionada sigue montada; no se apagó la unidad.'
        if not drive.get('CanPowerOff'): return 'USB desmontado. La unidad no admite apagado USB.'
        try:
            self.call(current.drive, DRIVE, 'PowerOff', [options()])
        except ExportError as exc:
            return 'USB desmontado; no se confirmó el apagado. '+str(exc)
        with self._lock: self._generation += 1
        self.changed.emit()
        return 'USB desmontado y apagado correctamente.'


if __name__ == '__main__':
    # Explicit read-only operator probe: no mount/unmount/power calls.
    import argparse
    from PyQt5.QtCore import QCoreApplication
    parser = argparse.ArgumentParser(description='Inspección USB de solo lectura mediante QtDBus/UDisks2')
    parser.add_argument('--inspect', action='store_true', help='Consultar dispositivos; no montar ni expulsar')
    args = parser.parse_args()
    if not args.inspect:
        parser.print_help(); raise SystemExit(0)
    app = QCoreApplication(sys.argv)
    try:
        provider = UDisksProvider([Path(__file__).resolve()])
        for destination in provider.discover():
            print(destination.label, destination.filesystem, 'montado' if destination.mount else 'sin montar')
    except ExportError as error:
        print(str(error)); raise SystemExit(1)
