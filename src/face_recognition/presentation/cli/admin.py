"""Explicit local credential setup; PINs are read only from a hidden terminal prompt."""
import argparse
from datetime import datetime, timezone
from getpass import getpass
from pathlib import Path

from face_recognition.domain.models.attendance import AttendancePersistenceError
from face_recognition.infrastructure.persistence.attendance_database import AttendanceDatabase
from face_recognition.infrastructure.persistence.admin_credentials import SQLiteAdminCredentialRepository


def main(argv=None):
    parser = argparse.ArgumentParser(description='Configuración local del administrador')
    parser.add_argument('--attendance-db', type=Path, required=True)
    parser.add_argument('command', choices=('setup',))
    args = parser.parse_args(argv)
    try:
        database = AttendanceDatabase(args.attendance_db)
        database.initialize()
        credentials = SQLiteAdminCredentialRepository(database)
        if credentials.stamp() is not None:
            parser.exit(1, 'El administrador ya está configurado. No se modificó el PIN.\n')
        pin = getpass('Nuevo PIN administrativo (seis dígitos): ')
        confirmation = getpass('Confirmar PIN: ')
        try:
            credentials.setup(pin, confirmation, datetime.now(timezone.utc))
        finally:
            pin = confirmation = ''
    except (ValueError, AttendancePersistenceError, OSError, EOFError):
        parser.exit(1, 'No fue posible configurar: revisa el PIN, su confirmación y la base de datos.\n')
    print('Administrador configurado correctamente.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
