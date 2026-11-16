"""Offscreen administrator workflows; synthetic SQLite, no camera or power commands."""
from datetime import datetime, timezone
import importlib.util
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
import unittest
from unittest.mock import patch, Mock

AVAILABLE = importlib.util.find_spec('PyQt5') is not None
if AVAILABLE:
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PyQt5.QtCore import QTimer
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication, QLineEdit, QPushButton
    from face_recognition.application.attendance_service import AttendanceService
    from face_recognition.domain.models.attendance import AttendancePersistenceError
    from face_recognition.infrastructure.device import DeviceDiagnostics
    from face_recognition.infrastructure.persistence.attendance_database import AttendanceDatabase
    from face_recognition.infrastructure.persistence.attendance_repositories import SQLiteAttendanceSessionRepository, SQLiteAttendanceRecordRepository
    from face_recognition.presentation.gui.admin import AdminIntegration
    from face_recognition.presentation.gui.attendance import AttendanceIntegration
    from face_recognition.presentation.gui.window import KioskWindow, LifecycleState
    from tests.admin_fakes import configured_admin, TEST_PIN


@unittest.skipUnless(AVAILABLE,'Optional Qt dependency unavailable')
class GuiAdministrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])
        if os.name=='nt':
            from PyQt5.QtGui import QFontDatabase
            if not QFontDatabase().families():
                QFontDatabase.addApplicationFont(os.path.join(os.environ.get('WINDIR','C:/Windows'),'Fonts','segoeui.ttf'))

    def setUp(self):
        self.temp=TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.db=AttendanceDatabase(Path(self.temp.name)/'synthetic.sqlite3'); self.db.initialize()
        self.now=datetime(2026,11,16,tzinfo=timezone.utc)
        self.admin=configured_admin(self.db,clock=lambda:self.now); self.admin.logout()
        self.sessions=SQLiteAttendanceSessionRepository(self.db); self.records=SQLiteAttendanceRecordRepository(self.db)
        self.service=AttendanceService(self.admin.participants,self.sessions,self.records,administrator=self.admin)
        self.w=KioskWindow(persistent_attendance=True)
        self.attendance=AttendanceIntegration(self.w,self.service,recording_enabled=True)
        self.power=Mock(enabled=True)
        self.ui=AdminIntegration(self.w,self.admin,self.admin.settings,
            DeviceDiagnostics(self.temp.name,platform='win32'),power=self.power)
        self.w.show(); self.addCleanup(self.ui.finalize); self.addCleanup(self.w.close)

    def wait_for(self,predicate):
        until=monotonic()+5
        while not predicate() and monotonic()<until: QTest.qWait(10)
        self.assertTrue(predicate())

    def login(self,pin=TEST_PIN):
        self.w.open_admin()
        for digit in pin:self.w.enter_digit(digit)
        self.w.confirm_pin(); self.wait_for(lambda:self.ui.job is None)

    def respond_pin(self,values):
        def respond():
            dialog=self.app.activeModalWidget()
            for field,value in zip(dialog.findChildren(QLineEdit),values):field.setText(value)
            next(b for b in dialog.findChildren(QPushButton) if b.text()=='Confirmar').click()
        QTimer.singleShot(0,respond)

    def test_persistent_navigation_login_mask_clear_logout_preserves_session(self):
        self.w.authenticated=True; self.w.navigate('settings')
        self.assertIs(self.w.lifecycle_state,LifecycleState.ADMIN_LOGIN)
        self.assertNotIn('123456',self.w.pin_hint.text())
        self.login('123456'); self.assertFalse(self.ui.allowed()); self.assertEqual(self.w.pin,'')
        self.login(); self.assertIs(self.w.lifecycle_state,LifecycleState.ADMIN)
        opened=self.service.create_session('Synthetic',self.now.date()).session
        self.attendance.refresh_session()
        self.w.navigate('attendance'); self.assertFalse(self.ui.allowed())
        self.assertEqual(self.sessions.current().id,opened.id)
        self.w.navigate('settings'); self.assertIs(self.w.lifecycle_state,LifecycleState.ADMIN_LOGIN)

    def test_preferences_startup_persistence_failure_rollback_and_no_initial_write(self):
        self.admin.settings.update_theme('light'); self.admin.settings.update_standby_timeout(120)
        with patch.object(self.admin.settings,'update_theme',side_effect=AssertionError('No initialization write')):
            self.ui.apply_settings(self.admin.settings.get())
        self.assertEqual(self.w.theme,'light'); self.assertEqual(self.w.standby_timeout_seconds,120)
        self.login(); self.w.navigate('settings')
        self.w.appearance.setCurrentIndex(0); self.w.timeout_selector.setCurrentIndex(1)
        self.assertEqual(self.admin.settings.get().theme,'dark'); self.assertEqual(self.admin.settings.get().standby_timeout_seconds,60)
        with patch.object(self.admin,'update_theme',side_effect=AttendancePersistenceError('private')):
            self.w.appearance.setCurrentIndex(1)
        self.assertEqual(self.w.theme,'dark'); self.assertEqual(self.w.appearance.currentIndex(),0)
        self.assertNotIn('private',self.w.settings_message.text())

    def test_lockout_countdown_and_return_navigation_does_not_reset_attempts(self):
        self.login()
        for _ in range(5):self.admin.credentials.verify('817263',self.now)
        self.w.navigate('attendance');self.w.open_admin()
        self.assertFalse(self.w.pin_confirm.isEnabled());self.assertIn('segundos',self.w.pin_message.text())
        self.assertTrue(self.ui.lock_timer.isActive())
        self.w.navigate('attendance');self.w.open_admin();self.assertFalse(self.w.pin_confirm.isEnabled())

    def test_sensitive_dialog_starts_locked_and_stops_countdown_after_cancellation(self):
        self.login()
        for _ in range(5):self.admin.credentials.verify('817263',self.now)
        def inspect():
            dialog=self.app.activeModalWidget()
            confirm=next(b for b in dialog.findChildren(QPushButton) if b.text()=='Confirmar')
            self.assertFalse(confirm.isEnabled())
            dialog.reject()
        QTimer.singleShot(0,inspect)
        with patch.object(self.admin,'authorize',side_effect=AssertionError('No verification during lock')):
            self.assertIsNone(self.ui.reauthenticate('scope'))
        self.w.navigate('attendance')
        self.assertFalse(self.ui.lock_timer.isActive())

    def test_unconfigured_gui_no_demo_fallback_and_safe_exit(self):
        with self.db.transaction(write=True) as c:c.execute('DELETE FROM admin_credentials')
        self.ui.update_setup_state();self.w.open_admin()
        self.assertFalse(self.w.pin_confirm.isEnabled())
        for digit in '123456':self.w.enter_digit(digit)
        self.w.confirm_pin();self.assertFalse(self.ui.allowed());self.assertEqual(self.w.pin,'')
        self.w.close();self.assertIs(self.w.lifecycle_state,LifecycleState.SHUTDOWN)

    def test_change_pin_uses_keypad_dialog_and_requires_new_login(self):
        self.login();self.w.navigate('settings')
        self.respond_pin((TEST_PIN,'092847','092847'))
        self.ui.action('pin')
        self.assertIs(self.w.lifecycle_state,LifecycleState.ADMIN_LOGIN)
        self.assertFalse(self.ui.allowed());self.assertEqual(self.w.pin,'')
        self.login('092847');self.assertTrue(self.ui.allowed())

    def test_reauthentication_grant_and_cancel_no_authorization(self):
        self.login()
        QTimer.singleShot(0,lambda:self.app.activeModalWidget().reject())
        with patch.object(self.admin,'authorize',wraps=self.admin.authorize) as verify:
            self.assertIsNone(self.ui.reauthenticate('scope'));verify.assert_not_called()
        self.respond_pin((TEST_PIN,))
        token=self.ui.reauthenticate('scope');self.assertIsNotNone(token);token.require('scope')
        with self.assertRaises(PermissionError):token.require('scope')

    def test_global_confirmation_cancel_and_exact_counts_then_real_reauthentication(self):
        self.login();self.w.navigate('settings')
        self.admin.participants.add('p001',self.now)
        session=self.service.create_session('Synthetic',self.now.date()).session
        self.service.register('p001',session.id);self.service.close_session(session.id)
        with patch.object(self.ui,'confirmation',return_value=False) as confirm, patch.object(self.ui,'reauthenticate',side_effect=AssertionError('No PIN after cancellation')):
            self.ui.action('history')
            self.assertIn('Sesiones: 1 · Asistencias: 1 · Participantes: 1',confirm.call_args.args[1])
        self.assertEqual(len(self.sessions.list()),1)
        with patch.object(self.ui,'confirmation',return_value=True):
            self.respond_pin((TEST_PIN,));self.ui.action('history')
        self.assertFalse(self.sessions.list());self.assertEqual(len(self.admin.participants.list()),1)
        self.assertTrue(self.admin.configured())

    def test_diagnostics_timer_counts_and_participant_data_protected(self):
        self.login();self.w.navigate('system')
        self.assertTrue(self.ui.diagnostic_timer.isActive())
        self.assertIn('v4',self.w.diagnostics['Base de datos'].text())
        self.assertEqual(self.w.diagnostics['CPU'].text(),'No disponible')
        with patch.object(self.admin,'summary',side_effect=AssertionError('No recurring database counts')):
            self.ui.refresh_metrics()
        self.w.navigate('dashboard');self.assertFalse(self.ui.diagnostic_timer.isActive())
        self.w.navigate('attendance')
        with patch.object(self.ui,'confirmation',side_effect=AssertionError('No protected data')):self.ui.show_participants()

    def test_fake_power_confirmation_authorization_cleanup_preserves_open_session(self):
        self.login();self.w.navigate('settings')
        session=self.service.create_session('Synthetic',self.now.date()).session
        self.attendance.refresh_session()
        with patch.object(self.ui,'confirmation',return_value=False):self.ui.action('reboot')
        self.assertIsNone(self.ui.pending_power);self.power.execute.assert_not_called()
        with patch.object(self.ui,'confirmation',return_value=True):
            self.respond_pin((TEST_PIN,));self.ui.action('poweroff')
        self.assertIs(self.w.lifecycle_state,LifecycleState.SHUTDOWN)
        self.assertEqual(self.ui.pending_power,'poweroff')
        self.assertEqual(self.sessions.current().id,session.id)
        self.assertFalse(self.ui.diagnostic_timer.isActive());self.assertFalse(self.ui.lock_timer.isActive())
        self.power.execute.assert_not_called() # Executed only by entry point after event loop/worker cleanup.

    def test_shutdown_during_authentication_task_leaves_no_authenticated_state(self):
        self.w.open_admin()
        for digit in TEST_PIN:self.w.enter_digit(digit)
        self.w.confirm_pin();self.w.close()
        self.wait_for(lambda:self.ui.job is None)
        self.assertFalse(self.ui.allowed());self.assertFalse(self.w._admin_shutdown_finished is False)

    def test_entry_point_fails_closed_for_unconfigured_live_recording_and_obsolete_flag(self):
        from contextlib import redirect_stderr
        from io import StringIO
        from face_recognition.presentation.gui.__main__ import main
        with self.db.transaction(write=True) as c:c.execute('DELETE FROM admin_credentials')
        with patch('PyQt5.QtWidgets.QApplication',return_value=self.app), patch('face_recognition.presentation.gui.recognition.RecognitionIntegration',side_effect=AssertionError('No model before credentials')):
            with redirect_stderr(StringIO()),self.assertRaises(SystemExit):
                main(['--attendance-db',str(self.db.path),'--model-dir','unused','--enable-attendance-recording'])
        for flags in (['--enable-synthetic-session-deletion'],['--enable-system-power-actions']):
            with redirect_stderr(StringIO()),self.assertRaises(SystemExit):main(flags)

    def test_entry_point_power_dispatch_occurs_after_cleanup_and_failure_is_reported(self):
        from contextlib import redirect_stderr
        from io import StringIO
        from face_recognition.presentation.gui.__main__ import main
        for fails in (False,True):
            trace=[]; controllers=[]
            power=Mock(enabled=True)
            def execute(action):
                self.assertIn('recognition stopped',trace)
                trace.append(action)
                if fails:raise PermissionError('Synthetic permission failure')
            power.execute.side_effect=execute
            recognition=Mock()
            recognition.finalize.side_effect=lambda:trace.append('recognition stopped')
            def controller(*args,**kwargs):
                result=AdminIntegration(*args,**kwargs);result.pending_power='reboot'
                controllers.append(result);return result
            with patch('PyQt5.QtWidgets.QApplication',return_value=self.app), patch.object(QApplication,'exec_',return_value=0), \
                    patch('face_recognition.presentation.gui.admin.AdminIntegration',side_effect=controller), \
                    patch('face_recognition.infrastructure.device.SystemPowerController',return_value=power), \
                    patch('face_recognition.presentation.gui.recognition.RecognitionIntegration',return_value=recognition), \
                    patch('face_recognition.presentation.gui.__main__.sys.platform','linux'),redirect_stderr(StringIO()) as output:
                code=main(['--attendance-db',str(self.db.path),'--model-dir','unused','--enable-system-power-actions'])
            self.assertEqual(code,1 if fails else 0)
            power.execute.assert_called_once_with('reboot')
            if fails:self.assertIn('cerró de forma segura',output.getvalue())
            for item in controllers:item.window.close()
