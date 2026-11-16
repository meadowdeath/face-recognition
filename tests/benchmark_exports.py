"""Reproducible synthetic generation benchmark; no USB, GUI, models or real DBs.

python -m tests.benchmark_exports --records 10000
Memory is tracemalloc's peak Python allocation, not total RSS/native allocations.
"""
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from time import perf_counter
import tracemalloc

from face_recognition.domain.models.export import ReportFilter
from face_recognition.infrastructure.persistence.attendance_database import AttendanceDatabase, timestamp
from face_recognition.infrastructure.persistence.admin_credentials import SQLiteAdminCredentialRepository
from face_recognition.infrastructure.persistence.attendance_backup import snapshot
from face_recognition.infrastructure.persistence.attendance_reports import report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=int, default=10000)
    args = parser.parse_args()
    if not 1 <= args.records <= 1_000_000: parser.error('records must be 1..1000000')
    with TemporaryDirectory(prefix='faceid-synthetic-benchmark-') as directory:
        root = Path(directory); db = AttendanceDatabase(root/'synthetic.sqlite3'); db.initialize()
        now = datetime(2026, 11, 16, tzinfo=timezone.utc); stamp = timestamp(now)
        SQLiteAdminCredentialRepository(db).setup('049283', '049283', now)
        with db.transaction(write=True) as c:
            c.executemany('INSERT INTO participants VALUES (?,?)', ((f'p{i:03}', stamp) for i in range(1, 101)))
            for i in range((args.records+99)//100):
                cursor = c.execute('INSERT INTO attendance_sessions(name,opened_at,closed_at,name_key,session_date) VALUES (?,?,?,?,?)',
                    (f'Evento sintético {i}', stamp, stamp, f'evento sintético {i}', '2026-11-16'))
                session = cursor.lastrowid
                c.executemany('INSERT INTO attendance_records(person_id,attendance_session_id,registered_at) VALUES (?,?,?)',
                    ((f'p{j+1:03}', session, timestamp(now+timedelta(seconds=i*100+j))) for j in range(min(100, args.records-i*100))))
        cancel = Event(); quiet = lambda *args: None
        source = root/'snapshot.sqlite3'; source.touch(mode=0o600)
        snapshot(db, source, cancel, quiet)
        print('format,records,seconds,bytes,peak_python_MiB')
        for format in ('CSV', 'XLSX', 'SQLite'):
            output = root/('result.'+('sqlite3' if format == 'SQLite' else format.lower()))
            output.touch(mode=0o600)
            tracemalloc.start(); started = perf_counter()
            if format == 'SQLite': snapshot(db, output, cancel, quiet)
            else: report(source, output, ReportFilter(), format, cancel, quiet)
            elapsed = perf_counter()-started; peak = tracemalloc.get_traced_memory()[1]; tracemalloc.stop()
            print(f'{format},{args.records},{elapsed:.3f},{output.stat().st_size},{peak/2**20:.2f}')


if __name__ == '__main__': main()
