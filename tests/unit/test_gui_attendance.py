"""Synthetic end-to-end attendance UI; no real model, camera or biometric data."""

from contextlib import redirect_stderr
from dataclasses import replace
from datetime import timedelta
from io import StringIO
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Event
from time import monotonic
import unittest
from unittest.mock import patch

from tests.admin_fakes import TEST_PIN, configured_admin
from face_recognition.application.attendance_service import AttendanceService
from face_recognition.domain.models.attendance import AttendancePersistenceError
from face_recognition.infrastructure.persistence.attendance_database import AttendanceDatabase
from face_recognition.infrastructure.persistence.attendance_repositories import (
    SQLiteParticipantRepository, SQLiteAttendanceSessionRepository, SQLiteAttendanceRecordRepository,
)

AVAILABLE = importlib.util.find_spec('PyQt5') is not None
if AVAILABLE:
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PyQt5.QtCore import QDate, QTimeZone, QTimer, Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication, QScrollArea, QLineEdit, QDateEdit, QLabel, QPushButton
    from face_recognition.domain.models.attendance import DeletionOperation
    from face_recognition.application.live_recognition import LiveRecognitionView
    from face_recognition.presentation.gui.admin import AdminIntegration
    from face_recognition.infrastructure.device import DeviceDiagnostics
    from face_recognition.presentation.gui.attendance import AttendanceIntegration, local_day_range
    from face_recognition.presentation.gui.window import KioskWindow, LifecycleState
    from face_recognition.presentation.gui.recognition import RecognitionIntegration, RecognitionWorker, LivePacket
    from face_recognition.presentation.gui.__main__ import main
    from tests.unit.test_gui_recognition import Components, options


@unittest.skipUnless(AVAILABLE, 'Optional PyQt5 dependency not installed')
class GuiAttendanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        if os.name == 'nt':
            from PyQt5.QtGui import QFontDatabase
            if not QFontDatabase().families():
                QFontDatabase.addApplicationFont(os.path.join(os.environ.get('WINDIR', 'C:/Windows'), 'Fonts', 'segoeui.ttf'))

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = AttendanceDatabase(Path(self.temp.name) / 'synthetic.sqlite3', busy_timeout_ms=10)
        self.db.initialize()
        self.people = SQLiteParticipantRepository(self.db)
        self.sessions = SQLiteAttendanceSessionRepository(self.db)
        self.records = SQLiteAttendanceRecordRepository(self.db)
        self.day = QDate(2026, 1, 2)
        start, end = local_day_range(self.day)
        self.now = start + (end - start) / 2
        self.administrator = configured_admin(self.db, clock=lambda: self.now)
        self.service = AttendanceService(self.people, self.sessions, self.records, clock=lambda: self.now, administrator=self.administrator)
        self.people.add('p001', self.now)
        self.people.add('p002', self.now)

    def window(self, *, live=False, recording=True, failed=False, deletion=False):
        w = KioskWindow(live_mode=live, persistent_attendance=True)
        attendance = AttendanceIntegration(w, None if failed else self.service,
            recording_enabled=recording, initialization_error='Base de datos no disponible.' if failed else '')
        AdminIntegration(w, self.administrator, self.administrator.settings,
                         DeviceDiagnostics(Path(self.temp.name), platform='win32'))
        if deletion:
            # Existing scope/dialog tests isolate reauthentication; the real PIN flow
            # is exercised separately by administration UI tests.
            w.administration.reauthenticate = lambda scope: self.administrator.authorize(scope, TEST_PIN)[1]
        w.show()
        def cleanup():
            w.close()
            self.app.processEvents()
        self.addCleanup(cleanup)
        return w, attendance

    def wait_for(self, predicate):
        deadline = monotonic() + 3
        while not predicate() and monotonic() < deadline: QTest.qWait(10)
        self.assertTrue(predicate())

    def live(self, *, recording=True, blocked=True):
        self.administrator.authenticate(TEST_PIN); self.opened = self.service.open_session('Synthetic live')
        self.w, self.attendance = self.window(live=True, recording=recording)
        # Reuse existing inference-only fixtures; these labels are fictional metadata.
        with patch('tests.live_open_set_fakes.CLASSES', ('p001', 'p002', 'p003')):
            self.components = Components()
        if blocked: self.components.read_gate = Event()
        self.worker = RecognitionWorker(options(preview_fps=0), components=self.components)
        self.live_ui = RecognitionIntegration(self.w, self.worker.options, worker=self.worker)
        def cleanup():
            if self.components.read_gate: self.components.read_gate.set()
            self.live_ui.finalize()
            self.w._shutdown_finished = True
        self.addCleanup(cleanup)
        self.w.navigate('attendance')
        self.wait_for(self.components.read_entered.is_set)

    def decision(self, sequence=1, **changes):
        view = LiveRecognitionView('p001', -12., -18., 0, 7, None,
            frame_candidate='p001', temporal_candidate='p001', temporal_decision='KNOWN',
            decision_sequence=sequence, decision_at=monotonic(), decision_is_current=True)
        return replace(view, **changes)

    def deliver(self, view, *, presence=True, generation=None):
        generation = self.live_ui.generation if generation is None else generation
        self.worker.publish(LivePacket(generation, view, presence))
        self.live_ui.drain()

    def test_persistent_demo_open_register_duplicate_unknown_unregistered_close_and_restart(self):
        w, attendance = self.window()
        self.assertIs(w.lifecycle_state, LifecycleState.STANDBY)
        self.assertIn('SINTÉTICA', w.mode_label.text())
        self.assertEqual(self.records.count_for_session(1), 0)
        w.navigate('attendance')
        self.assertEqual(w.attendance_status.text(), 'No hay una sesión de asistencia activa')
        w.session_name.setText('Synthetic class')
        self.administrator.authenticate(TEST_PIN); w.session_start.click()
        session_id = w.attendance_session_id
        self.assertIsNotNone(session_id)
        self.assertTrue(w.session_start.isEnabled())
        self.assertTrue(w.session_close.isEnabled())
        w.simulate_button.click()
        self.assertEqual(w.attendance_status.text(), 'Asistencia registrada')
        original_message = w.attendance_message.text()
        self.now += timedelta(minutes=1)
        w.simulate_button.click()
        self.assertEqual(w.attendance_status.text(), 'Asistencia ya registrada')
        self.assertEqual(w.attendance_message.text(), original_message)
        w.state_selector.setCurrentIndex(1)
        w.simulate_button.click()
        self.assertEqual(w.attendance_status.text(), 'Persona no reconocida')
        w.state_selector.setCurrentIndex(2)
        w.simulate_button.click()
        self.assertEqual(w.attendance_status.text(), 'Participante no registrado')
        self.assertFalse(self.people.exists('p003'))
        self.assertEqual(self.records.count_for_session(session_id), 1)
        w.close()  # Exit does not close the persistent session.
        restored, restored_attendance = self.window()
        self.assertIs(restored.lifecycle_state, LifecycleState.STANDBY)
        self.assertEqual(restored.attendance_session_id, session_id)
        self.assertIn('Synthetic class', restored.session_details.text())
        self.assertIn('Asistencias: 1', restored.session_details.text())
        self.administrator.authenticate(TEST_PIN); restored.session_close.click()
        self.assertIsNone(restored.attendance_session_id)
        self.assertFalse(self.sessions.get(session_id).is_open)
        self.assertEqual(self.records.count_for_session(session_id), 1)
        self.assertTrue(restored.session_start.isEnabled())

    def test_records_date_refresh_empty_and_closed_history_without_polling(self):
        self.administrator.authenticate(TEST_PIN); session = self.service.open_session('Historical synthetic')
        self.service.register('p001', session.id)
        self.administrator.authenticate(TEST_PIN); self.service.close_session(session.id)
        w, _ = self.window()
        self.administrator.authenticate(TEST_PIN)
        with patch.object(self.service, 'records_between', wraps=self.service.records_between) as query:
            w.records_date.setDate(self.day)
            w.navigate('records')
            self.assertEqual(w.records_table.rowCount(), 1)
            self.assertEqual(w.records_table.item(0, 0).text(), 'p001')
            self.assertEqual(w.records_table.item(0, 1).text(), f'Historical synthetic · {session.session_date:%d/%m/%Y}')
            count = query.call_count
            QTest.qWait(80)
            self.assertEqual(query.call_count, count)
            w.records_date.setDate(self.day.addDays(1))
            self.assertEqual(w.records_table.rowCount(), 0)
            self.assertIn('Sin registros', w.records_summary.text())

    def test_failures_never_fall_back_or_show_false_success(self):
        w, attendance = self.window(failed=True)
        self.assertFalse(w.session_start.isEnabled())
        w.set_session_open(True)
        self.assertFalse(w.session_open)
        w.navigate('attendance')
        self.assertEqual(w.attendance_status.text(), 'No fue posible registrar la asistencia')
        w.simulate_button.click()
        self.assertEqual(w.attendance_status.text(), 'No fue posible registrar la asistencia')
        w2, controller = self.window()
        self.administrator.authenticate(TEST_PIN); w2.session_start.click()
        self.assertIsNone(w2.attendance_session_id)
        self.assertIn('nombre', w2.session_hint.text())
        w2.session_name.setText('Synthetic')
        with patch.object(self.service, 'create_session', side_effect=AttendancePersistenceError('PRIVATE')):
            self.administrator.authenticate(TEST_PIN); w2.session_start.click()
        self.assertFalse(w2.session_open)
        self.assertNotIn('PRIVATE', w2.session_hint.text())
        self.administrator.authenticate(TEST_PIN); w2.session_start.click()
        w2.navigate('attendance')
        with self.db.transaction(write=True):
            w2.simulate_button.click()
        self.assertEqual(w2.attendance_status.text(), 'No fue posible registrar la asistencia')
        self.assertEqual(self.records.count_for_session(w2.attendance_session_id), 0)
        self.administrator.authenticate(TEST_PIN); self.service.close_session(w2.attendance_session_id)
        w2.simulate_button.click()
        self.assertEqual(w2.attendance_status.text(), 'No hay una sesión de asistencia activa')
        self.assertIsNone(w2.attendance_session_id)
        with patch.object(self.service, 'records_between', side_effect=AttendancePersistenceError('PRIVATE')):
            controller.refresh_records()
        self.assertIn('No fue posible consultar', w2.records_summary.text())

    def test_accepted_sequence_consumed_once_duplicate_preserves_time_and_confirmation(self):
        self.live()
        with patch.object(self.service, 'register', wraps=self.service.register) as register:
            first = self.decision()
            self.deliver(first)
            self.assertEqual(self.w.attendance_status.text(), 'Asistencia registrada')
            message = self.w.attendance_message.text()
            for _ in range(10): self.deliver(replace(first, occupancy=3, revision=100))
            self.assertEqual(register.call_count, 1)
            self.assertEqual(self.w.attendance_message.text(), message)
            self.now += timedelta(minutes=1)
            self.deliver(self.decision(sequence=2))
            self.assertEqual(register.call_count, 2)
            self.assertEqual(self.w.attendance_status.text(), 'Asistencia ya registrada')
            self.assertEqual(self.w.attendance_message.text(), message)
            self.assertEqual(self.records.count_for_session(self.opened.id), 1)
            # Even if the no-face packet was coalesced away, the invalidated
            # segment metadata clears the old confirmation without a new write.
            self.deliver(self.decision(sequence=2, decision_is_current=False))
            self.assertIsNone(self.attendance.confirmation)
            self.assertEqual(register.call_count, 2)
            self.deliver(self.decision(sequence=2, decision_is_current=False), presence=False)
            self.assertIsNone(self.attendance.confirmation)
            self.deliver(self.decision(sequence=2))
            self.assertEqual(register.call_count, 2)

    def test_incomplete_rejected_expired_and_cached_segment_decisions_never_write(self):
        self.live()
        with patch.object(self.service, 'register', wraps=self.service.register) as register:
            self.deliver(self.decision(sequence=0, label='COLLECTING', occupancy=6, temporal_decision=None))
            self.deliver(self.decision(sequence=1, label='UNKNOWN', temporal_decision='UNKNOWN'))
            self.assertEqual(self.w.attendance_status.text(), 'Persona no reconocida')
            self.deliver(self.decision(sequence=2, decision_at=monotonic()-10))
            self.deliver(self.decision(sequence=3, decision_is_current=False), presence=False)
            self.deliver(self.decision(sequence=3, decision_is_current=False))
            register.assert_not_called()
            self.assertEqual(self.records.count_for_session(self.opened.id), 0)
            self.deliver(self.decision(sequence=4))
            self.assertEqual(register.call_count, 1)

    def test_late_generation_and_previous_session_do_not_register(self):
        self.live()
        old_generation = self.live_ui.generation
        old_session = self.opened.id
        old_view = self.decision()
        self.w.open_admin()
        self.administrator.authenticate(TEST_PIN); self.attendance.close_session()
        self.w.session_name.setText('New session')
        self.administrator.authenticate(TEST_PIN); self.attendance.open_session()
        self.w.navigate('attendance')
        self.assertNotEqual(self.w.attendance_session_id, old_session)
        self.assertNotEqual(self.live_ui.generation, old_generation)
        with patch.object(self.service, 'register', wraps=self.service.register) as register:
            self.deliver(old_view, generation=old_generation)
            # Also reject a wrong session binding even with a current activation.
            self.assertFalse(self.attendance.handle_live(old_view, self.live_ui.generation,
                old_session, True, monotonic(), 2))
            register.assert_not_called()
        self.assertEqual(self.records.count_for_session(self.w.attendance_session_id), 0)

    def test_external_session_closure_prevents_write_and_deactivates_capture(self):
        self.live()
        self.administrator.authenticate(TEST_PIN); self.service.close_session(self.opened.id)
        self.deliver(self.decision())
        self.assertEqual(self.records.count_for_session(self.opened.id), 0)
        self.assertFalse(self.w.session_open)
        self.assertFalse(self.live_ui.active)

    def test_live_without_opt_in_keeps_identity_only_and_does_not_call_service(self):
        self.live(recording=False)
        with patch.object(self.service, 'register', side_effect=AssertionError('No opt-in')) as register:
            self.deliver(self.decision())
            register.assert_not_called()
        self.assertEqual(self.w.attendance_status.text(), 'Identidad reconocida')
        self.assertEqual(self.records.count_for_session(self.opened.id), 0)

    def test_fake_worker_end_to_end_commits_temporal_decision_only(self):
        self.live(blocked=False)
        self.wait_for(lambda: self.records.count_for_session(self.opened.id) == 1)
        self.assertEqual(self.records.list_for_session(self.opened.id)[0].person_id, 'p002')
        self.wait_for(lambda: self.w.attendance_status.text() in ('Asistencia registrada', 'Asistencia ya registrada'))
        self.assertEqual(len(self.components.loads), 1)
        self.w.open_admin()
        self.wait_for(lambda: self.components.active_cameras == 0)
        self.assertIsNone(self.attendance.confirmation)

    def test_theme_layout_and_persistent_settings_preserve_unrelated_preferences(self):
        w, _ = self.window()
        for theme in ('dark', 'light'):
            w.apply_theme(theme)
            for name, page in w.screens.items():
                w.stack.setCurrentWidget(page)
                self.app.processEvents()
                self.assertLessEqual(page.minimumSizeHint().height(), 444, name)
                self.assertLessEqual(page.minimumSizeHint().width(), 625, name)
                if name == 'sessions':
                    scroll = page.findChild(QScrollArea)
                    self.assertLessEqual(scroll.widget().minimumSizeHint().width(), scroll.viewport().width())
        w.timeout_selector.setCurrentIndex(2)
        self.assertEqual(w.standby_timeout_seconds, 120)
        from face_recognition.infrastructure.persistence.attendance_repositories import SQLiteDeviceSettingsRepository
        self.assertEqual(SQLiteDeviceSettingsRepository(self.db).get().theme, 'dark')
        self.assertEqual(SQLiteDeviceSettingsRepository(self.db).get().standby_timeout_seconds, 120)

    def test_local_calendar_ranges_handle_dst_and_midnight_edges(self):
        zone = QTimeZone(b'America/New_York')
        if not zone.isValid(): self.skipTest('Qt timezone database unavailable')
        for day, hours in ((QDate(2026, 3, 8), 23), (QDate(2026, 11, 1), 25)):
            start, end = local_day_range(day, zone)
            self.assertEqual((end-start).total_seconds(), hours * 3600)
            self.assertIsNotNone(start.utcoffset())
            self.now = start
            self.administrator.authenticate(TEST_PIN); session = self.service.open_session('Synthetic boundary ' + day.toString('yyyy-MM-dd'))
            self.service.register('p001', session.id)
            self.now = end
            self.service.register('p002', session.id)
            rows = self.service.records_between(start, end)
            self.assertEqual([r.person_id for r, _ in rows], ['p001'])
            self.administrator.authenticate(TEST_PIN); self.service.close_session(session.id)

    def test_cli_opt_in_validation_and_persistent_demo_imports_no_recognition(self):
        for flags in (['--enable-attendance-recording'], ['--enable-attendance-recording', '--attendance-db', str(self.db.path)],
                      ['--enable-attendance-recording', '--model-dir', 'unused']):
            with self.subTest(flags=flags), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                main(flags)
        path = Path(self.temp.name) / 'entry.sqlite3'
        script = '''
import sys
from PyQt5.QtWidgets import QApplication
from face_recognition.presentation.gui.__main__ import main
QApplication.exec_ = lambda self: 0
assert main(['--windowed', '--attendance-db', sys.argv[1]]) == 0
assert not any(name in sys.modules for name in ('numpy', 'cv2', 'mediapipe', 'sklearn',
    'face_recognition.presentation.gui.recognition'))
'''
        result = subprocess.run([sys.executable, '-c', script, str(path)], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(path.exists())
        self.assertIsNone(SQLiteAttendanceSessionRepository(AttendanceDatabase(path)).current())

    def test_initialization_failure_entry_does_not_fall_back(self):
        path = Path(self.temp.name) / 'corrupt.sqlite3'
        path.write_bytes(b'not sqlite')
        def inspect():
            window = next(w for w in QApplication.topLevelWidgets() if isinstance(w, KioskWindow)
                          and w.persistent_attendance and w.isVisible())
            self.assertIsNone(window.attendance.service)
            self.assertFalse(window.session_start.isEnabled())
            self.assertFalse(window.session_open)
            window.close()
            return 0
        with patch('PyQt5.QtWidgets.QApplication', return_value=self.app), \
                patch.object(QApplication, 'exec_', side_effect=inspect):
            self.assertEqual(main(['--windowed', '--attendance-db', str(path)]), 0)
        self.assertEqual(path.read_bytes(), b'not sqlite')

    def test_explicit_creation_selection_resumption_counts_and_restart(self):
        w, controller = self.window()
        w.session_name.setText('Optimización - 2026-11-16')
        self.administrator.authenticate(TEST_PIN); w.session_start.click()
        identifier = w.attendance_session_id
        self.service.register('p001', identifier, 1)
        controller.refresh_session()
        self.assertIn('Sesión creada correctamente', w.session_hint.text())
        w.session_name.setText(' OPTIMIZACIÓN - 2026-11-16 ')
        self.administrator.authenticate(TEST_PIN); w.session_start.click()
        self.assertIn('Ya existe una sesión con ese nombre y fecha', w.session_hint.text())
        self.assertEqual(w.attendance_activation_revision, 1)
        self.assertFalse(w.session_resume.isEnabled())
        self.assertFalse(w.session_edit.isEnabled())
        w.session_name.setText('Different')
        self.administrator.authenticate(TEST_PIN); w.session_start.click()
        self.assertIn('Cierra la sesión activa', w.session_hint.text())
        self.assertEqual(w.attendance_session_id, identifier)
        self.administrator.authenticate(TEST_PIN); w.session_close.click()
        self.assertTrue(w.session_resume.isEnabled())
        self.assertTrue(w.session_edit.isEnabled())
        w.session_resume.click()
        self.assertEqual(w.attendance_session_id, identifier)
        self.assertEqual(w.attendance_activation_revision, 2)
        self.assertEqual(controller.count, 1)
        self.assertIn('Asistencias: 1', w.session_details.text())
        self.assertEqual(w.session_hint.text(), 'Sesión reanudada correctamente.')
        self.administrator.authenticate(TEST_PIN); w.session_close.click()
        w.session_name.setText('optimización - 2026-11-16')
        self.administrator.authenticate(TEST_PIN); w.session_start.click()
        self.assertIsNone(w.attendance_activation_revision)
        self.assertIn('Ya existe una sesión con ese nombre y fecha', w.session_hint.text())
        w.session_resume.click()
        self.assertEqual(w.attendance_activation_revision, 3)
        self.assertEqual(len(self.sessions.list()), 1)
        w.close()
        restored, _ = self.window()
        self.assertIs(restored.lifecycle_state, LifecycleState.STANDBY)
        self.assertEqual(restored.attendance_session_id, identifier)
        self.assertEqual(restored.attendance_activation_revision, 3)
        self.assertIn('Asistencias: 1', restored.session_details.text())

    def test_session_edit_dialog_confirmation_collision_and_filter_refresh(self):
        self.administrator.authenticate(TEST_PIN); first = self.service.open_session('First')
        self.service.register('p001', first.id)
        self.administrator.authenticate(TEST_PIN); self.service.close_session(first.id)
        self.administrator.authenticate(TEST_PIN); other = self.service.open_session('Other')
        self.administrator.authenticate(TEST_PIN); self.service.close_session(other.id)
        w, controller = self.window()
        w.stored_sessions.setCurrentIndex(w.stored_sessions.findData(first.id))
        w.records_session.setCurrentIndex(w.records_session.findData(first.id))
        self.assertTrue(w.session_edit.isEnabled())
        def respond(name, accepted):
            dialog = self.app.activeModalWidget()
            self.assertEqual(dialog.windowTitle(), 'Editar sesión')
            self.assertEqual(dialog.findChild(QLineEdit, 'edit_session_name').text(), 'First')
            self.assertEqual(dialog.styleSheet(), w.styleSheet())
            dialog.findChild(QLineEdit, 'edit_session_name').setText(name)
            dialog.accept() if accepted else dialog.reject()
        for name, accepted, expected in (('Cancelled', False, 'First'), ('OTHER', True, 'First'), ('Renamed', True, 'Renamed')):
            QTimer.singleShot(0, lambda n=name, a=accepted: respond(n, a))
            w.session_edit.click()
            self.assertEqual(self.sessions.get(first.id).name, expected)
        self.assertEqual(w.stored_sessions.currentData(), first.id)
        self.assertEqual(w.records_session.currentData(), first.id)
        self.assertEqual(w.records_session.currentText(), f'Renamed · {first.session_date:%d/%m/%Y}')
        w.records_date.setDate(self.day)
        self.administrator.authenticate(TEST_PIN)
        w.navigate('records')
        self.assertEqual(w.records_table.item(0, 1).text(), f'Renamed · {first.session_date:%d/%m/%Y}')
        self.assertEqual(self.records.count_for_session(first.id), 1)

    def test_session_and_date_filters_full_history_and_unrestricted_guard(self):
        self.administrator.authenticate(TEST_PIN); first = self.service.open_session('First')
        self.service.register('p001', first.id)
        self.now += timedelta(days=1)
        self.service.register('p002', first.id)
        self.administrator.authenticate(TEST_PIN); self.service.close_session(first.id)
        self.administrator.authenticate(TEST_PIN); second = self.service.open_session('Second')
        self.service.register('p001', second.id)
        w, controller = self.window()
        self.administrator.authenticate(TEST_PIN)
        w.records_date.setDate(self.day)
        w.navigate('records')
        self.assertTrue(w.records_use_date.isChecked())
        self.assertEqual(w.records_table.rowCount(), 1)
        w.records_session.setCurrentIndex(w.records_session.findData(first.id))
        self.assertEqual(w.records_table.rowCount(), 1)
        w.records_date.setDate(self.day.addDays(1))
        self.assertEqual(w.records_table.rowCount(), 1)
        self.assertEqual(w.records_table.item(0, 0).text(), 'p002')
        w.records_use_date.setChecked(False)
        self.assertFalse(w.records_date.isEnabled())
        self.assertEqual(w.records_table.rowCount(), 2)
        self.assertIn('/', w.records_table.item(0, 2).text())
        self.assertNotEqual(w.records_table.item(0, 2).text(), w.records_table.item(1, 2).text())
        with patch.object(self.service, 'filtered_records', wraps=self.service.filtered_records) as query:
            w.records_session.setCurrentIndex(0)
            self.assertEqual(w.records_table.rowCount(), 0)
            self.assertIn('Selecciona una sesión', w.records_summary.text())
            query.assert_not_called()
        w.records_session.setCurrentIndex(w.records_session.findData(second.id))
        self.assertEqual(w.records_table.rowCount(), 1)
        w.records_use_date.setChecked(True)
        w.records_date.setDate(self.day.addDays(3))
        self.assertEqual(w.records_table.rowCount(), 0)
        self.assertIn('Sin registros', w.records_summary.text())

    def test_same_session_resumption_invalidates_worker_generation_and_revision_binding(self):
        self.live()
        identifier, generation = self.w.attendance_session_id, self.live_ui.generation
        first_view = self.decision()
        self.administrator.authenticate(TEST_PIN); self.service.close_session(identifier, 1)
        self.administrator.authenticate(TEST_PIN); self.assertEqual(self.service.resume_session(identifier).session.activation_revision, 2)
        # UI still holds activation 1: the service must refuse it even though the ID is open again.
        self.deliver(first_view)
        self.assertEqual(self.records.count_for_session(identifier), 0)
        self.assertEqual(self.w.attendance_activation_revision, 2)
        self.assertNotEqual(self.live_ui.generation, generation)
        self.assertIsNone(self.attendance.confirmation)
        self.deliver(first_view, generation=generation)
        self.assertEqual(self.records.count_for_session(identifier), 0)
        self.assertFalse(self.attendance.handle_live(self.decision(), self.live_ui.generation,
            identifier, True, monotonic(), 2, activation_revision=1))
        self.deliver(self.decision())
        self.assertEqual(self.records.count_for_session(identifier), 1)

    def test_refresh_detects_same_id_new_revision_and_clears_confirmation(self):
        self.live()
        self.deliver(self.decision())
        self.assertIsNotNone(self.attendance.confirmation)
        previous = self.live_ui.generation
        self.administrator.authenticate(TEST_PIN); self.service.close_session(self.opened.id)
        self.administrator.authenticate(TEST_PIN); self.service.resume_session(self.opened.id)
        self.attendance.refresh_session()
        self.assertGreater(self.live_ui.generation, previous)
        self.assertIsNone(self.attendance.confirmation)
        self.assertEqual(self.live_ui.session_revision, 2)

    def dated_events(self):
        events = []
        for name, day, person in (('Optimización', self.day.toPyDate(), 'p001'),
                                  ('optimización', self.day.addDays(7).toPyDate(), 'p002'),
                                  ('Inglés', self.day.toPyDate(), 'p001')):
            self.administrator.authenticate(TEST_PIN); session = self.service.create_session(name, day).session
            self.service.register(person, session.id)
            self.administrator.authenticate(TEST_PIN); self.service.close_session(session.id)
            events.append(self.sessions.get(session.id))
        return events

    def select_event(self, w, event):
        w.stored_names.setCurrentIndex(w.stored_names.findData(event.name_key))
        w.stored_sessions.setCurrentIndex(w.stored_sessions.findData(event.id))

    def test_dated_creation_tabs_linked_selectors_and_edit_date(self):
        w, controller = self.window()
        self.assertEqual([w.session_tabs.tabText(i) for i in range(3)], ['Crear sesión', 'Gestionar sesiones', 'Eliminar sesiones'])
        self.assertEqual(w.session_create_date.date(), QDate.currentDate())
        self.assertEqual(w.session_start.text(), 'Crear e iniciar sesión')
        w.session_name.setText('Optimización')
        w.session_create_date.setDate(self.day)
        self.administrator.authenticate(TEST_PIN); w.session_start.click()
        first = self.sessions.current()
        self.service.register('p001', first.id)
        self.administrator.authenticate(TEST_PIN); w.session_close.click()
        self.administrator.authenticate(TEST_PIN); w.session_start.click()
        self.assertIn('Ya existe una sesión con ese nombre y fecha', w.session_hint.text())
        self.assertIsNone(w.attendance_session_id)
        w.session_create_date.setDate(self.day.addDays(7))
        self.administrator.authenticate(TEST_PIN); w.session_start.click()
        second = self.sessions.current()
        self.administrator.authenticate(TEST_PIN); w.session_close.click()
        self.assertEqual(w.stored_names.count(), 1)
        self.assertEqual(w.stored_sessions.count(), 2)
        self.select_event(w, first)
        def edit():
            dialog = self.app.activeModalWidget()
            dialog.findChild(QLineEdit, 'edit_session_name').setText('Aplicada')
            dialog.findChild(QDateEdit, 'edit_session_date').setDate(self.day.addDays(1))
            dialog.accept()
        QTimer.singleShot(0, edit)
        w.session_edit.click()
        actual = self.sessions.get(first.id)
        self.assertEqual(actual.session_date, self.day.addDays(1).toPyDate())
        self.assertEqual(actual.opened_at, first.opened_at)
        self.assertEqual(self.records.get(first.id, 'p001').registered_at, self.now)
        self.assertEqual(w.stored_names.count(), 2)
        self.assertEqual(w.stored_sessions.currentData(), first.id)
        self.assertIn('Aplicada', w.records_session.itemText(w.records_session.findData(first.id)))
        self.assertEqual(self.sessions.get(second.id).session_date, second.session_date)

    def test_destructive_dialog_cancel_close_escape_and_all_scope_text(self):
        first, second, other = self.dated_events()
        w, controller = self.window(deletion=True)
        self.select_event(w, first)
        expected = (
            ('Vaciar asistencias de esta sesión', 'Vaciar asistencias', 0, 1),
            ('Eliminar sesión de esta fecha', 'Eliminar sesión', 1, 1),
            ('Eliminar sesiones de todas las fechas', 'Eliminar todas las fechas', 2, 2),
        )
        for mode in ('cancel', 'close', 'escape'):
            for index, (title, confirm, sessions, records) in enumerate(expected):
                def respond(i=index, t=title, c=confirm, sc=sessions, rc=records):
                    dialog = self.app.activeModalWidget()
                    self.assertEqual(dialog.windowTitle(), t)
                    text = dialog.findChild(QLabel, 'deletion_summary').text()
                    self.assertIn(first.name, text)
                    self.assertIn(first.session_date.strftime('%d/%m/%Y'), text)
                    self.assertIn(f'Sesiones que se eliminarán: {sc}', text)
                    self.assertIn(f'Asistencias que se eliminarán: {rc}', text)
                    self.assertIn('participantes y las preferencias se conservarán', text)
                    self.assertIn('no se puede deshacer', text)
                    if i == 2:
                        self.assertIn(second.session_date.strftime('%d/%m/%Y'), text)
                        self.assertIn('Todas las fechas', text)
                    buttons = {b.text(): b for b in dialog.findChildren(QPushButton)}
                    self.assertTrue(buttons['Cancelar'].isDefault())
                    self.assertFalse(buttons[c].isDefault())
                    self.assertEqual(len(self.sessions.list()), 3)
                    if mode == 'cancel': buttons['Cancelar'].click()
                    elif mode == 'close': dialog.close()
                    else: QTest.keyClick(dialog, Qt.Key_Escape)
                QTimer.singleShot(0, respond)
                w.deletion_buttons[index].click()
                self.assertEqual(len(self.sessions.list()), 3)
                self.assertEqual(sum(c for _, c in self.sessions.list_with_counts()), 3)

    def test_all_three_confirmed_operations_refresh_scopes_and_records(self):
        first, second, other = self.dated_events()
        w, controller = self.window(deletion=True)
        self.select_event(w, first)
        w.records_date.setDate(self.day)
        w.records_session.setCurrentIndex(w.records_session.findData(first.id))
        def accept(): self.app.activeModalWidget().accept()
        QTimer.singleShot(0, accept)
        w.deletion_buttons[0].click()
        self.assertIsNotNone(self.sessions.get(first.id))
        self.assertEqual(self.records.count_for_session(first.id), 0)
        self.assertEqual(w.records_table.rowCount(), 0)
        self.assertEqual(w.stored_sessions.currentData(), first.id)
        self.assertFalse(w.deletion_buttons[0].isEnabled())
        QTimer.singleShot(0, accept)
        w.deletion_buttons[1].click()
        self.assertIsNone(self.sessions.get(first.id))
        self.assertEqual(w.stored_sessions.findData(first.id), -1)
        self.assertEqual(w.records_session.findData(first.id), -1)
        self.select_event(w, second)
        QTimer.singleShot(0, accept)
        w.deletion_buttons[2].click()
        self.assertEqual(w.stored_names.findData(first.name_key), -1)
        self.assertEqual(self.records.count_for_session(other.id), 1)
        self.assertEqual(len(self.people.list()), 2)
        self.assertIsNotNone(self.sessions.get(other.id))
        self.assertIn('Sesiones eliminadas: 1 · Asistencias eliminadas: 1', w.session_hint.text())

    def test_changed_confirmation_aborts_even_when_record_count_is_unchanged(self):
        first, _, _ = self.dated_events()
        w, _ = self.window(deletion=True)
        self.select_event(w, first)
        def change_and_accept():
            record = self.records.get(first.id, 'p001')
            with self.db.transaction(write=True) as c:
                c.execute('DELETE FROM attendance_records WHERE id=?', (record.id,))
                from face_recognition.infrastructure.persistence.attendance_database import timestamp
                c.execute('INSERT INTO attendance_records (person_id,attendance_session_id,registered_at) VALUES (?,?,?)',
                          ('p002', first.id, timestamp(record.registered_at)))
            self.app.activeModalWidget().accept()
        QTimer.singleShot(0, change_and_accept)
        w.deletion_buttons[1].click()
        self.assertIsNotNone(self.sessions.get(first.id))
        self.assertEqual(self.records.count_for_session(first.id), 1)
        self.assertIn('datos cambiaron', w.session_hint.text())

    def test_deletion_requires_actual_administrator_session_including_live_mode(self):
        first, _, _ = self.dated_events()
        for live in (False, True):
            w, controller = self.window(live=live, deletion=True)
            self.select_event(w, first)
            self.assertTrue(w.session_tabs.isTabEnabled(2))
            self.administrator.logout()
            controller.delete_selected(DeletionOperation.SESSION)
            self.assertIn('No se puede eliminar', w.session_hint.text())
            self.administrator.authenticate(TEST_PIN)
        self.assertEqual(len(self.sessions.list()), 3)

    def test_deletion_is_disabled_and_revalidated_while_any_session_is_open(self):
        first, _, _ = self.dated_events()
        w, controller = self.window(deletion=True)
        self.administrator.authenticate(TEST_PIN); self.service.create_session('Active unrelated', self.day.toPyDate())
        controller.refresh_session()
        self.select_event(w, first)
        self.assertTrue(all(not b.isEnabled() for b in w.deletion_buttons))
        controller.delete_selected(DeletionOperation.NAME)
        self.assertEqual(len(self.sessions.list()), 4)
        self.assertIn('cierra la sesión activa', w.session_hint.text())

    def test_synthetic_deletion_flag_combinations_are_rejected_before_initialization(self):
        for flags in (['--enable-synthetic-session-deletion'],
                      ['--enable-synthetic-session-deletion','--attendance-db',str(self.db.path),'--model-dir','unused'],
                      ['--enable-synthetic-session-deletion','--attendance-db',str(self.db.path),'--enable-attendance-recording'],
                      ['--enable-synthetic-session-deletion','--attendance-db',str(self.db.path),'--model-dir','unused','--enable-attendance-recording']):
            with self.subTest(flags=flags), redirect_stderr(StringIO()), self.assertRaises(SystemExit): main(flags)
