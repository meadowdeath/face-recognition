"""Local participant registration and name editing."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from face_recognition.domain.models.attendance import AttendancePersistenceError, normalize_participant_name, validate_person_id
from face_recognition.infrastructure.persistence.attendance_database import AttendanceDatabase
from face_recognition.infrastructure.persistence.attendance_repositories import SQLiteParticipantRepository


def main(argv=None):
    parser = argparse.ArgumentParser(description="Alta explícita de identificadores; no acredita consentimiento")
    parser.add_argument("--attendance-db", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("add", "update"):
        action = commands.add_parser(command)
        action.add_argument("person_id")
        action.add_argument("--given-names", required=command == "add")
        action.add_argument("--surnames", required=command == "add")
    commands.add_parser("list")
    args = parser.parse_args(argv)
    if args.command == "update" and args.given_names is None and args.surnames is None:
        parser.error("update requires --given-names or --surnames")
    try:
        if args.command != "list":
            validate_person_id(args.person_id)
            names = {
                field: normalize_participant_name(value)
                for field, value in (("given_names", args.given_names), ("surnames", args.surnames))
                if value is not None
            }
        database = AttendanceDatabase(args.attendance_db, busy_timeout_ms=150)
        database.initialize()
        participants = SQLiteParticipantRepository(database)
        if args.command == "add":
            participants.add(args.person_id, datetime.now(timezone.utc), **names)
            print(f"Participante registrado: {args.person_id}")
        elif args.command == "update":
            participants.update_names(args.person_id, **names)
            print(f"Participante actualizado: {args.person_id}")
        else:
            print("ID del participante | Nombre(s) | Apellidos")
            for participant in participants.list():
                print(f"{participant.person_id} | {participant.given_names or '—'} | {participant.surnames or '—'}")
    except (ValueError, LookupError, AttendancePersistenceError, OSError):
        parser.exit(1, "No fue posible completar la operación. Comprueba el identificador, nombres y la base de datos.\n")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
