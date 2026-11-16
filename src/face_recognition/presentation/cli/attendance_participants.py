"""Explicit technical participant registration; never discover IDs from a model."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from face_recognition.domain.models.attendance import AttendancePersistenceError, validate_person_id
from face_recognition.infrastructure.persistence.attendance_database import AttendanceDatabase
from face_recognition.infrastructure.persistence.attendance_repositories import SQLiteParticipantRepository


def main(argv=None):
    parser = argparse.ArgumentParser(description="Alta explícita de identificadores; no acredita consentimiento")
    parser.add_argument("--attendance-db", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("add").add_argument("person_id")
    commands.add_parser("list")
    args = parser.parse_args(argv)
    try:
        if args.command == "add":
            validate_person_id(args.person_id)
        database = AttendanceDatabase(args.attendance_db, busy_timeout_ms=150)
        database.initialize()
        participants = SQLiteParticipantRepository(database)
        if args.command == "add":
            participants.add(args.person_id, datetime.now(timezone.utc))
            print(f"Participante registrado: {args.person_id}")
        else:
            for participant in participants.list():
                print(participant.person_id)
    except (ValueError, AttendancePersistenceError, OSError):
        parser.exit(1, "No fue posible completar la operación. Comprueba el identificador, duplicados y la base de datos.\n")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
