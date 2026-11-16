"""Exclusive, directory-anchored USB publication. No overwrite fallback on Linux."""
import ctypes
import errno
import os
from pathlib import Path
import shutil
import stat
import sys
from time import monotonic
from uuid import uuid4

from face_recognition.domain.models.export import ExportError
from .attendance_backup import check_cancel

FAT32_LIMIT = 2**32 - 1


def verify_directory(descriptor, destination):
    if os.fstat(descriptor).st_dev != destination.device:
        raise ExportError('replacement', 'El directorio USB pertenece a otro montaje.')
    if sys.platform.startswith('linux'):
        fields = dict(line.split(':', 1) for line in Path(f'/proc/self/fdinfo/{descriptor}').read_text().splitlines())
        if fields.get('mnt_id', '').strip() != destination.mount_id:
            raise ExportError('replacement', 'El montaje del directorio USB cambió.')


def rename_exclusive(directory, source, target, path):
    if sys.platform.startswith('linux'):
        libc = ctypes.CDLL(None, use_errno=True)
        rename = getattr(libc, 'renameat2', None)
        if rename is None:
            raise ExportError('publication', 'El sistema no admite publicación segura sin sobrescritura.')
        rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
        rename.restype = ctypes.c_int
        if rename(directory, os.fsencode(source), directory, os.fsencode(target), 1):
            number = ctypes.get_errno()
            if number == errno.EEXIST:
                raise ExportError('collision', 'El archivo ya existe; prepara una nueva exportación.')
            raise ExportError('publication', 'El USB no admite publicación segura sin sobrescritura.')
    elif os.name == 'nt':
        # Windows rename fails if the destination exists; unlike POSIX rename.
        os.rename(path/source, path/target)
    else:
        raise ExportError('publication', 'Publicación segura no disponible en esta plataforma.')


def publish(staged, request, provider, cancel, progress):
    destination = provider.validate(request.destination)
    check_cancel(cancel, monotonic()+300)
    if not destination.mount:
        raise ExportError('unmounted', 'Monta el USB antes de exportar.')
    if destination.read_only:
        raise ExportError('readonly', 'El USB es de solo lectura.')
    size = Path(staged).stat().st_size
    if destination.filesystem in ('vfat', 'fat', 'fat32') and size > FAT32_LIMIT:
        raise ExportError('size', 'El archivo excede el límite de FAT32 (4 GiB menos un byte).')
    mount = Path(destination.mount)
    parts = ('FaceID', 'respaldos' if request.operation == 'backup' else 'registros')
    temporary = '.faceid_'+uuid4().hex+'.tmp'
    descriptors, directory, path, created, published = [], None, mount, False, False
    deadline = monotonic()+300
    anchored = os.name != 'nt'
    try:
        if mount.is_symlink() or (getattr(mount.lstat(), 'st_file_attributes', 0)
                                & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0)):
            raise ExportError('path', 'Ruta USB insegura.')
        if anchored:
            directory = os.open(mount, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            descriptors.append(directory)
            verify_directory(directory, destination)
        for part in parts:
            provider.validate(destination)
            if anchored:
                try: os.mkdir(part, mode=0o700, dir_fd=directory)
                except FileExistsError: pass
                directory = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
                descriptors.append(directory)
                verify_directory(directory, destination)
            else:
                path = path/part
                if path.is_symlink() or (path.exists() and getattr(path.lstat(), 'st_file_attributes', 0)
                                         & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0)):
                    raise ExportError('path', 'Ruta USB insegura.')
                path.mkdir(mode=0o700, exist_ok=True)
        if anchored:
            space = os.fstatvfs(directory)
            free = space.f_bavail * space.f_frsize
        else:
            free = shutil.disk_usage(path).free
        if free < size + 1024*1024:
            raise ExportError('space', 'Espacio insuficiente en el USB.')
        provider.validate(destination)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0)
        fd = os.open(temporary if anchored else path/temporary, flags, 0o600,
                     **({'dir_fd': directory} if anchored else {}))
        created = True
        with os.fdopen(fd, 'wb') as output, open(staged, 'rb') as source:
            transferred = 0
            while data := source.read(1024*1024):
                check_cancel(cancel, deadline)
                output.write(data); transferred += len(data)
                progress('Escribiendo USB', int(transferred*100/max(size, 1)))
            try:
                output.flush(); os.fsync(output.fileno())
            except OSError as exc:
                raise ExportError('sync', 'No se pudo sincronizar el archivo USB; no se publicó el resultado.') from exc
            if os.fstat(output.fileno()).st_size != size:
                raise ExportError('write', 'No se escribió el archivo completo.')
        check_cancel(cancel, deadline)
        provider.validate(destination)
        check_cancel(cancel, deadline)
        rename_exclusive(directory, temporary, request.filename, path)
        published = True
        try:
            if anchored: os.fsync(directory)
            provider.validate(destination)
        except (OSError, ExportError) as exc:
            raise ExportError('durability', 'Archivo publicado: '+request.filename+
                              '. No se confirmó su durabilidad; verifica el USB antes de retirarlo.') from exc
        return size
    except OSError as exc:
        messages = {
            errno.EEXIST: ('collision', 'El archivo ya existe; prepara otra exportación.'),
            errno.EROFS: ('readonly', 'El USB es de solo lectura.'),
            errno.ENOSPC: ('space', 'Espacio insuficiente en el USB.'),
            errno.EACCES: ('permission', 'No hay permisos para escribir en el USB.'),
            errno.EPERM: ('permission', 'No hay permisos para escribir en el USB.'),
            errno.ELOOP: ('path', 'Ruta USB insegura: no se permiten enlaces simbólicos.'),
            errno.ENOTDIR: ('path', 'Ruta USB insegura o no válida.'),
        }
        code, message = messages.get(exc.errno, ('write', 'La escritura USB falló; revisa la conexión y el dispositivo.'))
        raise ExportError(code, message) from exc
    finally:
        if created and not published:
            try:
                if anchored: os.unlink(temporary, dir_fd=directory)
                else: (path/temporary).unlink()
            except OSError as exc:
                for descriptor in reversed(descriptors): os.close(descriptor)
                descriptors.clear()
                raise ExportError('partial', 'No se pudo retirar el archivo parcial '+temporary+'. Verifica el USB.') from exc
        for descriptor in reversed(descriptors): os.close(descriptor)
