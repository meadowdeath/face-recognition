"""Portable report, full-backup, authorization and cancellation regressions."""
from contextlib import closing
from dataclasses import replace
from datetime import timedelta, timezone
import csv
import importlib.util
import unittest
from pathlib import Path
import sqlite3
from unittest.mock import patch

from face_recognition.domain.models.export import ExportError, ReportFilter
from face_recognition.infrastructure.persistence.attendance_backup import snapshot, validate_backup
from face_recognition.infrastructure.persistence.attendance_reports import csv_literal, report
from tests.export_fakes import ExportFixture, NOW


class AttendanceExportTests(ExportFixture):
    def test_csv_bom_headers_quotes_unicode_timestamp_order_and_no_credentials(self):
        preview = self.service.prepare('report', 'CSV', self.provider.current)
        result = self.execute(preview.request)
        path = self.usb/'FaceID/registros'/result.filename
        self.assertTrue(path.read_bytes().startswith(b'\xef\xbb\xbf'))
        with path.open(encoding='utf-8-sig', newline='') as stream: rows = list(csv.reader(stream))
        self.assertEqual(tuple(rows[0]), ('ID de asistencia', 'ID del participante', 'Nombre', 'Apellidos',
                                        'ID de sesión', 'Nombre de sesión', 'Fecha de sesión',
                                        'Fecha de registro', 'Hora de registro', 'Registro UTC'))
        self.assertTrue(all(len(row) == 10 for row in rows))
        self.assertEqual(rows[1][2:4], ['Lúmina Alba', 'de Bruma'])
        self.assertEqual(result.rows, 3)
        self.assertEqual(rows[1][1], 'p001'); self.assertEqual(rows[1][5], 'Optimización, «A»')
        self.assertEqual(rows[3][5], "'=1+1")
        self.assertEqual(rows[1][6], '2026-11-01'); self.assertEqual(rows[1][9], '2026-11-16T00:00:00.000000Z')
        self.assertEqual(self.sessions.current(), self.second)
        self.assertNotIn('pin_hash', path.read_text(encoding='utf-8-sig'))

    def test_csv_formula_prefixes_whitespace_controls_full_width_and_quoting(self):
        for text in ('=1', '+2', '-3', '@x', ' \t=1', '\r\n+1', '＝1', '＋1', '\ufeff＠x'):
            self.assertEqual(csv_literal(text), "'"+text)
        self.assertEqual(csv_literal('Inglés, "A"'), 'Inglés, "A"')
        self.assertEqual(csv_literal('p001'), 'p001')

    def test_filters_use_actual_registration_range_and_session_id(self):
        filters = ReportFilter(self.first.id, NOW, NOW+timedelta(days=1))
        preview = self.service.prepare('report', 'CSV', self.provider.current, filters)
        self.assertEqual(preview.records, 1)
        self.assertEqual(self.execute(preview.request).rows, 1)
        self.assertEqual(self.service.prepare('report', 'CSV', self.provider.current,
                         ReportFilter(self.first.id)).records, 2)
        with self.assertRaises(ValueError): ReportFilter(start=NOW, end=NOW)
        with self.assertRaises(ValueError): ReportFilter(start=NOW)

    @unittest.skipUnless(importlib.util.find_spec('openpyxl'), 'Optional XLSX dependency unavailable')
    def test_xlsx_reopens_literal_text_types_freeze_filters_and_private_spool_cleanup(self):
        from openpyxl import load_workbook
        result = self.execute(self.service.prepare('report', 'XLSX', self.provider.current).request)
        path = self.usb/'FaceID/registros'/result.filename
        with closing(load_workbook(path)) as book:
            sheet = book['Asistencias']
            self.assertEqual(sheet.max_row, 4); self.assertEqual(sheet.freeze_panes, 'A2')
            self.assertEqual(sheet.auto_filter.ref, 'A1:J4')
            self.assertEqual(sheet['F4'].value, '=1+1'); self.assertEqual(sheet['F4'].data_type, 's')
            self.assertEqual(sheet['B2'].value, 'p001'); self.assertEqual(sheet['B2'].data_type, 's')
            self.assertEqual(sheet.max_column, 10)
            self.assertEqual((sheet['C2'].value, sheet['D2'].value), ('Lúmina Alba', 'de Bruma'))
            self.assertEqual((sheet['C2'].data_type, sheet['D2'].data_type), ('s', 's'))
            self.assertTrue(sheet['A1'].font.bold)
            self.assertEqual(sheet['J2'].value, '2026-11-16T00:00:00.000000Z')
            self.assertEqual(sheet['G2'].value.date().isoformat(), '2026-11-01')
        self.assertFalse(list(self.usb.rglob('*.xml')))

    def test_legacy_name_cells_are_empty_and_export_defends_imported_text(self):
        with self.db.transaction(write=True) as c:
            c.execute("UPDATE participants SET given_names=NULL, surnames=NULL WHERE person_id='p002'")
            c.execute("UPDATE participants SET given_names=? WHERE person_id='p001'", ('＝1+1',))
        result = self.execute(self.service.prepare('report', 'CSV', self.provider.current).request)
        with (self.usb/'FaceID/registros'/result.filename).open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.reader(stream))
        self.assertEqual(rows[1][2], "'＝1+1")
        self.assertEqual(rows[2][2:4], ['', ''])

    @unittest.skipUnless(importlib.util.find_spec('openpyxl'), 'Optional XLSX dependency unavailable')
    def test_xlsx_treats_imported_formula_name_as_literal_and_preserves_missing_names(self):
        from openpyxl import load_workbook
        with self.db.transaction(write=True) as c:
            c.execute("UPDATE participants SET given_names=NULL, surnames=NULL WHERE person_id='p002'")
            c.execute("UPDATE participants SET given_names=? WHERE person_id='p001'", ('=1+1',))
        result = self.execute(self.service.prepare('report', 'XLSX', self.provider.current).request)
        with closing(load_workbook(self.usb/'FaceID/registros'/result.filename)) as book:
            sheet = book['Asistencias']
            self.assertEqual((sheet['C2'].value, sheet['C2'].data_type), ('=1+1', 's'))
            self.assertIsNone(sheet['C3'].value)
            self.assertIsNone(sheet['D3'].value)

    def test_empty_report_fails_without_publishing_and_invalidates_grant(self):
        request = self.service.prepare('report', 'CSV', self.provider.current,
                                      ReportFilter(start=NOW-timedelta(days=2), end=NOW-timedelta(days=1))).request
        with self.assertRaisesRegex(ExportError, 'No hay'): self.execute(request)
        self.assertFalse((self.usb/'FaceID').exists())

    def test_preview_is_estimate_and_snapshot_has_final_consistent_count(self):
        request = self.service.prepare('report', 'CSV', self.provider.current).request
        self.records.add('p002', self.second.id, NOW+timedelta(days=3))
        self.assertEqual(self.execute(request).rows, 4)

    def test_report_reads_snapshot_not_later_source_changes(self):
        path = self.root/'snapshot.sqlite3'
        snapshot(self.db, path, self.cancel, lambda *args: None)
        self.records.add('p002', self.second.id, NOW+timedelta(days=3))
        self.assertEqual(report(path, self.root/'out.csv', ReportFilter(), 'CSV', self.cancel,
                                lambda *args: None, zone=timezone(timedelta(hours=-6))), 3)
        with (self.root/'out.csv').open(encoding='utf-8-sig', newline='') as stream: rows = list(csv.reader(stream))
        self.assertEqual(rows[1][7:9], ['2026-11-15', '18:00:00'])

    def test_complete_backup_compares_every_table_sequences_lockout_and_open_session(self):
        with self.db.transaction(write=True) as c:
            c.execute("UPDATE admin_credentials SET failed_attempts=5,locked_until='2026-11-16T00:01:00.000000Z'")
            c.execute("UPDATE sqlite_sequence SET seq=100 WHERE name='attendance_sessions'")
        path = self.root/'complete.sqlite3'
        snapshot(self.db, path, self.cancel, lambda *args: None)
        validate_backup(path)
        with self.db.transaction() as source, closing(sqlite3.connect(path)) as target:
            for table in ('participants', 'attendance_sessions', 'attendance_records', 'device_settings', 'admin_credentials', 'sqlite_sequence'):
                self.assertEqual([tuple(r) for r in source.execute('SELECT * FROM '+table+' ORDER BY 1')],
                                 target.execute('SELECT * FROM '+table+' ORDER BY 1').fetchall())
            self.assertEqual(target.execute('PRAGMA user_version').fetchone()[0], 5)
            self.assertEqual(target.execute('PRAGMA foreign_key_check').fetchall(), [])
        self.assertTrue(self.sessions.current().is_open)

    def test_authorized_backup_is_independent_of_report_filters(self):
        preview = self.service.prepare('backup', 'SQLite', self.provider.current, ReportFilter(self.first.id))
        self.assertEqual(preview.request.filters, ReportFilter())
        result = self.execute(preview.request)
        self.assertTrue(result.integrity_verified)
        validate_backup(self.usb/'FaceID/respaldos'/result.filename)

    def test_backup_cancellation_deadline_corruption_and_concurrent_source_activity(self):
        self.cancel.set()
        with self.assertRaises(ExportError): snapshot(self.db, self.root/'cancel.sqlite3', self.cancel, lambda *args: None)
        self.cancel.clear()
        with self.assertRaises(ExportError): snapshot(self.db, self.root/'timeout.sqlite3', self.cancel, lambda *args: None, seconds=0)
        def progress(*args):
            if not self.admin.participants.exists('p003'): self.admin.participants.add('p003', NOW, given_names="Lúmina Alba", surnames="de Bruma")
        snapshot(self.db, self.root/'live.sqlite3', self.cancel, progress)
        validate_backup(self.root/'live.sqlite3')
        with closing(sqlite3.connect(self.root/'live.sqlite3')) as c:
            c.execute('PRAGMA user_version=99')
        with self.assertRaises(ExportError): validate_backup(self.root/'live.sqlite3')

    def test_grants_reject_operation_format_filters_filename_usb_reconnect_and_reuse(self):
        request = self.service.prepare('report', 'CSV', self.provider.current).request
        changes = (dict(operation='backup', format='SQLite', filename='backup.sqlite3'),
                   dict(format='XLSX', filename='report.xlsx'), dict(filters=ReportFilter(self.first.id)),
                   dict(filename='different.csv'))
        for change in changes:
            permit = self.permitted(request)
            with self.assertRaises(PermissionError): self.service.execute(replace(request, **change), permit, self.cancel, lambda *a: None)
        permit = self.permitted(request)
        self.provider.current = replace(self.provider.current, identity='reconnected')
        with self.assertRaises(ExportError): self.service.execute(request, permit, self.cancel, lambda *a: None)
        self.provider.current = request.destination
        result = self.execute(request)
        with self.assertRaises(PermissionError): self.service.execute(request, permit, self.cancel, lambda *a: None)
        self.assertEqual(result.rows, 3)

    def test_authentication_expiry_logout_cancellation_and_busy_fail_closed(self):
        request = self.service.prepare('report', 'CSV', self.provider.current).request
        permit = self.permitted(request); self.admin.timer = lambda: float('inf')
        with self.assertRaises(PermissionError): self.service.execute(request, permit, self.cancel, lambda *a: None)
        self.admin.logout()
        with self.assertRaises(PermissionError): self.service.prepare('backup', 'SQLite', self.provider.current)
        self.admin.authenticate('049283')
        permit = self.permitted(request); self.cancel.set()
        with self.assertRaises(ExportError): self.service.execute(request, permit, self.cancel, lambda *a: None)
        self.cancel.clear()
        self.service._operation.acquire()
        try:
            with self.assertRaises(ExportError): self.service.execute(request, permit, self.cancel, lambda *a: None)
        finally: self.service._operation.release()
        self.assertFalse((self.usb/'FaceID').exists())

    def test_failed_export_removes_private_staging_and_never_publishes(self):
        request = self.service.prepare('report', 'CSV', self.provider.current).request
        staging = []
        def fail(snapshot, *args):
            staging.append(Path(snapshot).parent); raise OSError('synthetic failure')
        with patch.object(self.service, 'writer', side_effect=fail), self.assertRaises(OSError): self.execute(request)
        self.assertTrue(staging); self.assertFalse(staging[0].exists())
        self.assertFalse((self.usb/'FaceID').exists())

    def test_success_consumes_grant_and_another_usb_cannot_reuse_it(self):
        request = self.service.prepare('report', 'CSV', self.provider.current).request
        permit = self.permitted(request)
        self.service.execute(request, permit, self.cancel, lambda *a: None)
        with self.assertRaises(PermissionError): self.service.execute(request, permit, self.cancel, lambda *a: None)
        permit = self.permitted(request)
        self.provider.current = replace(request.destination, identity='other-usb')
        changed = replace(request, destination=self.provider.current)
        with self.assertRaises(PermissionError): self.service.execute(changed, permit, self.cancel, lambda *a: None)

    def test_cancel_during_online_backup_and_source_lock_are_bounded(self):
        target = self.root/'cancel-midway.sqlite3'
        def cancelled(*args): self.cancel.set()
        with self.assertRaises(ExportError): snapshot(self.db, target, self.cancel, cancelled)
        self.cancel.clear()
        from face_recognition.infrastructure.persistence.attendance_database import AttendanceDatabase
        bounded = AttendanceDatabase(self.db.path, busy_timeout_ms=10)
        with closing(sqlite3.connect(self.db.path, isolation_level=None)) as locked:
            locked.execute('BEGIN EXCLUSIVE')
            try:
                with self.assertRaises(sqlite3.OperationalError): snapshot(bounded, self.root/'locked.sqlite3', self.cancel, lambda *a: None)
            finally: locked.rollback()
        self.assertEqual(self.records.count_for_session(self.first.id), 2)

    def test_xlsx_row_limit_is_explicit(self):
        from unittest.mock import MagicMock
        connection = MagicMock()
        connection.execute.return_value.fetchone.return_value = (1_048_576,)
        with patch('face_recognition.infrastructure.persistence.attendance_reports.sqlite3.connect', return_value=connection):
            with self.assertRaisesRegex(ExportError, 'límite'):
                report(self.db.path, self.root/'limit.xlsx', ReportFilter(), 'XLSX', self.cancel, lambda *a: None)
        self.assertFalse((self.root/'limit.xlsx').exists())
