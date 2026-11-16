"""Offscreen administrator workflows; synthetic SQLite, no camera or power commands."""
from datetime import datetime, timezone, timedelta
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
    from PyQt5.QtCore import QTimer, Qt, QDate, QTime, QDateTime, QPoint
    from PyQt5.QtGui import QPalette
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication, QLineEdit, QPushButton, QLabel, QDateEdit, QTimeEdit, QSlider, QScrollArea, QTableWidget, QTabWidget, QComboBox, QFrame, QTableWidgetItem
    from face_recognition.application.attendance_service import AttendanceService
    from face_recognition.domain.models.attendance import AttendancePersistenceError
    from face_recognition.infrastructure.device import DeviceDiagnostics
    from face_recognition.infrastructure.persistence.attendance_database import AttendanceDatabase
    from face_recognition.infrastructure.persistence.attendance_repositories import SQLiteAttendanceSessionRepository, SQLiteAttendanceRecordRepository
    from face_recognition.presentation.gui.admin import AdminIntegration
    from face_recognition.presentation.gui.attendance import AttendanceIntegration
    from face_recognition.presentation.gui.window import KioskWindow, LifecycleState
    from face_recognition.presentation.gui.themes import THEMES
    from face_recognition.infrastructure.persistence.attendance_repositories import SQLiteDeviceSettingsRepository
    from tests.admin_fakes import use_fast_credentials, configured_admin, TEST_PIN


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
        use_fast_credentials(self)
        from tests.export_fakes import unavailable_usb
        self.enterContext(patch('face_recognition.infrastructure.udisks.UDisksProvider', new=unavailable_usb))
        self.dialog_errors=[]
        self.addCleanup(lambda:self.assertEqual(self.dialog_errors,[]))
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

    def schedule_dialog(self, callback):
        def respond():
            dialog=self.app.activeModalWidget()
            try:callback(dialog)
            except Exception as exc:
                self.dialog_errors.append(f'{type(exc).__name__}: {exc}')
                dialog.busy=False
                dialog.reject()
        QTimer.singleShot(0,respond)

    @staticmethod
    def dialog_button(dialog,text):
        return next(b for b in dialog.findChildren(QPushButton) if b.text()==text)

    def type_pin(self,dialog,field,value):
        QTest.mouseClick(field,Qt.LeftButton)
        self.app.processEvents()
        for digit in value:self.dialog_button(dialog,digit).click()

    def respond_pin(self,values):
        def respond(dialog):
            for field,value in zip(dialog.findChildren(QLineEdit),values):
                self.type_pin(dialog,field,value)
            self.dialog_button(dialog,'Confirmar').click()
        self.schedule_dialog(respond)

    def login_keypad(self,pin=TEST_PIN):
        self.w.open_admin()
        page=self.w.screens['pin']
        for digit in pin:self.dialog_button(page,digit).click()
        self.dialog_button(page,'Confirmar').click()
        self.wait_for(lambda:self.ui.job is None)

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
        for _ in range(5):self.admin.credentials.verify('817263',lambda:self.now)
        self.w.navigate('attendance');self.w.open_admin()
        self.assertFalse(self.w.pin_confirm.isEnabled());self.assertIn('segundos',self.w.pin_message.text())
        self.assertTrue(self.ui.lock_timer.isActive())
        self.w.navigate('attendance');self.w.open_admin();self.assertFalse(self.w.pin_confirm.isEnabled())

    def test_sensitive_dialog_starts_locked_and_stops_countdown_after_cancellation(self):
        self.login()
        for _ in range(5):self.admin.credentials.verify('817263',lambda:self.now)
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

    def test_authentication_uses_one_worker_and_does_not_block_gui_or_survive_cancellation(self):
        from threading import Event, get_ident
        from tests.admin_fakes import fast_test_derive
        from face_recognition.infrastructure.persistence import admin_credentials
        entered,release=Event(),Event(); gui_thread=get_ident(); worker_threads=[]; ticks=[]
        def derive(*args):
            worker_threads.append(get_ident());entered.set()
            if not release.wait(5):raise AssertionError('Authentication was not released')
            return fast_test_derive(*args)
        self.w.open_admin()
        with patch.object(admin_credentials,'derive',side_effect=derive):
            try:
                for digit in TEST_PIN:self.w.enter_digit(digit)
                self.w.confirm_pin();first=self.ui.job
                self.wait_for(entered.is_set)
                QTimer.singleShot(0,lambda:ticks.append(True));self.wait_for(lambda:bool(ticks))
                self.w.confirm_pin();self.assertIs(self.ui.job,first)
                self.assertFalse(self.w.pin_confirm.isEnabled())
                self.w.open_admin()  # Revisiting login must not re-enable a pending task.
                self.assertFalse(self.w.pin_confirm.isEnabled())
                self.w.navigate('attendance')
            finally:release.set()
            self.wait_for(lambda:self.ui.job is None)
        self.assertEqual(len(worker_threads),1);self.assertNotEqual(worker_threads[0],gui_thread)
        self.assertFalse(self.ui.allowed());self.assertIs(self.w.lifecycle_state,LifecycleState.ATTENDANCE)

    def test_authentication_failure_debug_output_contains_timing_not_secrets(self):
        from contextlib import redirect_stdout
        from io import StringIO
        self.w.live_debug_enabled=True
        stamp=self.admin.credentials.stamp()
        with patch.object(self.admin.credentials,'verify',side_effect=AttendancePersistenceError('Private detail')),redirect_stdout(StringIO()) as output:
            self.login()
        self.assertFalse(self.ui.allowed());self.assertEqual(self.w.pin,'')
        self.assertIn('base de datos',self.w.pin_message.text())
        self.assertIn('ms',output.getvalue())
        for secret in (TEST_PIN,stamp[0].hex(),stamp[1].hex(),'Private detail'):
            self.assertNotIn(secret,output.getvalue())

    def test_shutdown_waits_for_blocked_authentication_task(self):
        from threading import Event
        from tests.admin_fakes import fast_test_derive
        from face_recognition.infrastructure.persistence import admin_credentials
        entered,release=Event(),Event()
        def derive(*args):
            entered.set()
            if not release.wait(5):raise AssertionError('Authentication was not released')
            return fast_test_derive(*args)
        self.w.open_admin()
        with patch.object(admin_credentials,'derive',side_effect=derive):
            try:
                for digit in TEST_PIN:self.w.enter_digit(digit)
                self.w.confirm_pin();self.wait_for(entered.is_set)
                self.w.close()
                self.assertIs(self.w.lifecycle_state,LifecycleState.SHUTDOWN)
                self.assertFalse(self.w._admin_shutdown_finished)
            finally:release.set()
            self.wait_for(lambda:self.ui.job is None)
        self.assertTrue(self.w._admin_shutdown_finished);self.assertFalse(self.ui.allowed())

    def test_keypad_digits_checked_signal_length_backspace_focus_and_masking(self):
        self.login_keypad()
        for theme in ('dark','light'):
            self.w.apply_theme(theme)
            before=self.admin.credentials.stamp()
            def interact(dialog):
                fields=dialog.findChildren(QLineEdit)
                self.assertEqual(len(fields),3)
                self.assertLessEqual(dialog.width(),800);self.assertLessEqual(dialog.height(),480)
                for index,field in enumerate(fields):
                    self.assertEqual(field.echoMode(),QLineEdit.Password)
                    self.assertEqual(field.maxLength(),6)
                    self.type_pin(dialog,field,('012345','678901','098765')[index])
                    self.assertEqual(field.text(),('012345','678901','098765')[index])
                    self.assertIs(dialog.active_field,field)
                    self.assertTrue(field.property('pinActive'),'Active field marker missing')
                    self.assertTrue(all(not f.property('pinActive') for f in fields if f is not field))
                    self.dialog_button(dialog,'8').click();self.assertEqual(len(field.text()),6)
                    self.dialog_button(dialog,'Borrar').click();self.assertEqual(len(field.text()),5)
                    # Explicitly emit the checked-state overload. It must never become the digit.
                    self.dialog_button(dialog,'2').clicked.emit(True)
                    self.assertEqual(field.text()[-1],'2')
                    self.assertTrue(field.hasFocus(),'Keypad changed Qt focus')
                    self.assertEqual(field.validator().validate('１２３４５６',0)[0],0)
                    self.assertEqual(field.validator().validate('12a456',0)[0],0)
                QTest.mouseClick(fields[1],Qt.LeftButton,pos=QPoint(fields[1].width()-8,fields[1].height()//2))
                self.app.processEvents()
                self.dialog_button(dialog,'Confirmar').setFocus()
                self.dialog_button(dialog,'Borrar').click()
                self.dialog_button(dialog,'3').click()
                self.assertEqual(fields[1].text()[-1],'3')
                self.assertIs(dialog.active_field,fields[1])
                self.dialog_button(dialog,'Cancelar').click()
            self.schedule_dialog(interact)
            self.assertIsNone(self.ui.pin_dialog(changing=True))
            self.assertEqual(self.admin.credentials.stamp(),before)

    def test_keypad_pin_change_logs_out_rejects_old_and_accepts_replacement(self):
        self.login_keypad();self.w.navigate('settings')
        before=self.admin.credentials.stamp()
        self.respond_pin((TEST_PIN,'092847','092847'))
        self.w.admin_actions['pin'].click()
        self.assertIs(self.w.lifecycle_state,LifecycleState.ADMIN_LOGIN)
        self.assertFalse(self.ui.allowed())
        self.assertNotEqual(self.admin.credentials.stamp()[1],before[1])
        self.login_keypad(TEST_PIN);self.assertFalse(self.ui.allowed())
        self.login_keypad('092847');self.assertTrue(self.ui.allowed())

    def test_keypad_wrong_current_and_mismatch_do_not_change_credentials(self):
        self.login_keypad();before=self.admin.credentials.stamp()
        for values,expected in ((('817263','092847','092847'),'PIN incorrecto'),
                                ((TEST_PIN,'092847','193847'),'confirmación')):
            def reject_after_response(dialog):
                for field,value in zip(dialog.findChildren(QLineEdit),values):self.type_pin(dialog,field,value)
                self.dialog_button(dialog,'Confirmar').click()
                self.wait_for(lambda:self.ui.job is None)
                self.assertTrue(all(f.text()=='' for f in dialog.findChildren(QLineEdit)))
                self.assertTrue(any(expected in label.text() for label in dialog.findChildren(QLabel)))
                self.dialog_button(dialog,'Cancelar').click()
            self.schedule_dialog(reject_after_response)
            self.assertIsNone(self.ui.pin_dialog(changing=True))
            self.assertEqual(self.admin.credentials.stamp(),before)

    def test_keypad_reauthentication_is_single_use_and_cancel_clears_inputs(self):
        self.login_keypad()
        self.respond_pin((TEST_PIN,))
        token=self.ui.reauthenticate('keypad-scope')
        self.assertIsNotNone(token);token.require('keypad-scope')
        with self.assertRaises(PermissionError):token.require('keypad-scope')
        with patch.object(self.admin,'authorize',wraps=self.admin.authorize) as authorize:
            def cancel(dialog):
                self.type_pin(dialog,dialog.findChildren(QLineEdit)[0],'012')
                self.dialog_button(dialog,'Cancelar').click()
            self.schedule_dialog(cancel)
            self.assertIsNone(self.ui.reauthenticate('cancelled-scope'))
            authorize.assert_not_called()

    def test_keypad_inputs_and_duplicate_submissions_disabled_while_verifying(self):
        from threading import Event
        from tests.admin_fakes import fast_test_derive
        from face_recognition.infrastructure.persistence import admin_credentials
        self.login_keypad();entered,release=Event(),Event()
        def derive(*args):
            entered.set()
            if not release.wait(5):raise AssertionError('Verification was not released')
            return fast_test_derive(*args)
        def interact(dialog):
            try:
                self.type_pin(dialog,dialog.findChildren(QLineEdit)[0],TEST_PIN)
                confirm=self.dialog_button(dialog,'Confirmar');confirm.click()
                self.wait_for(entered.is_set);job=self.ui.job
                self.assertTrue(dialog.busy)
                self.assertTrue(all(not f.isEnabled() and f.text()=='' for f in dialog.findChildren(QLineEdit)))
                for digit in '0123456789':
                    self.assertFalse(self.dialog_button(dialog,digit).isEnabled())
                    self.dialog_button(dialog,digit).click()
                confirm.clicked.emit(False)
                self.assertIs(self.ui.job,job)
                self.assertTrue(any('Verificando' in label.text() for label in dialog.findChildren(QLabel)))
            finally:release.set()
        self.schedule_dialog(interact)
        with patch.object(admin_credentials,'derive',side_effect=derive),patch.object(self.admin,'authorize',wraps=self.admin.authorize) as authorize:
            token=self.ui.reauthenticate('busy-scope')
            self.assertIsNotNone(token);authorize.assert_called_once()

    def seed_keypad_events(self):
        self.login_keypad()
        for person in ('p001','p002'):self.admin.participants.add(person,self.now)
        events=[]
        for name,day,person in (('Synthetic',self.now.date(),'p001'),
                               ('Synthetic',self.now.date()+timedelta(days=7),'p002'),
                               ('Other',self.now.date(),'p001')):
            session=self.service.create_session(name,day).session
            self.service.register(person,session.id);self.service.close_session(session.id)
            events.append(self.sessions.get(session.id))
        self.attendance.refresh_session()
        return events

    def confirm_then_keypad(self,check):
        def confirm(dialog):
            check(dialog)
            self.dialog_button(dialog,next(b.text() for b in dialog.findChildren(QPushButton) if b.objectName()=='destructive')).click()
            self.respond_pin((TEST_PIN,))
        self.schedule_dialog(confirm)

    def targeted_keypad_deletion(self,index):
        first,second,other=self.seed_keypad_events()
        self.w.navigate('sessions')
        self.w.stored_names.setCurrentIndex(self.w.stored_names.findData(first.name_key))
        self.w.stored_sessions.setCurrentIndex(self.w.stored_sessions.findData(first.id))
        self.w.session_tabs.setCurrentIndex(2)
        def check(dialog):
            text=dialog.findChild(QLabel,'deletion_summary').text()
            self.assertIn(first.session_date.strftime('%d/%m/%Y'),text)
            if index==2:self.assertIn(second.session_date.strftime('%d/%m/%Y'),text)
            self.assertTrue(self.dialog_button(dialog,'Cancelar').isDefault())
        self.confirm_then_keypad(check)
        self.w.deletion_buttons[index].click()
        self.assertEqual(self.records.count_for_session(first.id),0)
        self.assertEqual(self.sessions.get(first.id) is None,index!=0)
        self.assertEqual(self.sessions.get(second.id) is None,index==2)
        self.assertEqual(self.records.count_for_session(other.id),1)
        self.assertEqual(len(self.admin.participants.list()),2)

    def test_keypad_authorizes_clearing_targeted_attendance(self):
        self.targeted_keypad_deletion(0)

    def test_keypad_authorizes_deleting_one_session(self):
        self.targeted_keypad_deletion(1)

    def test_keypad_authorizes_deleting_all_dates_for_one_name(self):
        self.targeted_keypad_deletion(2)

    def global_keypad_reset(self,key):
        self.seed_keypad_events();self.w.navigate('settings')
        self.confirm_then_keypad(lambda dialog:self.assertTrue(any('Sesiones: 3' in item.text() for item in dialog.findChildren(QLabel))))
        self.w.admin_actions[key].click()
        self.assertEqual(len(self.sessions.list()),0)
        self.assertEqual(len(self.admin.participants.list()),0 if key=='admin_data' else 2)
        self.assertTrue(self.admin.configured())

    def test_keypad_authorizes_global_history_deletion(self):
        self.global_keypad_reset('history')

    def test_keypad_authorizes_global_administrative_data_reset(self):
        self.global_keypad_reset('admin_data')

    def test_preference_reset_shows_snapshot_values_and_uses_keypad(self):
        events=self.seed_keypad_events();self.w.navigate('settings')
        self.w.appearance.setCurrentIndex(1);self.w.timeout_selector.setCurrentIndex(2)
        def check(dialog):
            text=' '.join(label.text() for label in dialog.findChildren(QLabel))
            for value in ('Lavender (Claro)','120 segundos','Amethyst (Oscuro)','30 segundos'):
                self.assertIn(value,text)
        self.confirm_then_keypad(check)
        self.w.admin_actions['preferences'].click()
        self.assertEqual(self.admin.settings.get().theme,'dark')
        self.assertEqual(self.admin.settings.get().standby_timeout_seconds,30)
        self.assertEqual(len(self.sessions.list()),len(events));self.assertEqual(len(self.admin.participants.list()),2)

    def test_settings_clock_is_read_only_uses_footer_timer_and_does_not_touch_sqlite(self):
        before=QDateTime(QDate(2026,11,16),QTime(23,59,59),Qt.LocalTime)
        after=QDateTime(QDate(2026,11,17),QTime(0,0,0),Qt.LocalTime)
        with patch('face_recognition.presentation.gui.window.QDateTime') as clock,patch.object(self.db,'transaction',side_effect=AssertionError('Clock must not access SQLite')):
            for instant in (before,after):
                clock.currentDateTime.return_value=instant
                self.w.clock.timeout.emit()
                self.assertEqual(self.w.system_date.text(),'Fecha del sistema: '+instant.toString('dd/MM/yyyy'))
                self.assertEqual(self.w.system_time.text(),'Hora del sistema: '+self.w.time_label.text())
                self.assertEqual(self.w.date_label.text(),self.w.locale().toString(instant.date(),'ddd d MMM yyyy'))
        page=self.w.screens['settings']
        self.assertFalse(page.findChildren(QSlider));self.assertFalse(page.findChildren(QDateEdit));self.assertFalse(page.findChildren(QTimeEdit))
        self.assertIsInstance(self.w.system_date,QLabel);self.assertIsInstance(self.w.system_time,QLabel)
        self.assertTrue(self.w.session_create_date.isEnabled());self.assertTrue(self.w.records_date.isEnabled())
        self.assertFalse(self.w.session_create_date.isReadOnly());self.assertFalse(self.w.records_date.isReadOnly())

    def test_disabled_power_actions_have_explanation(self):
        window=KioskWindow(persistent_attendance=True)
        ui=AdminIntegration(window,self.admin,self.admin.settings,DeviceDiagnostics(self.temp.name,platform='win32'))
        self.addCleanup(ui.finalize);self.addCleanup(window.close)
        self.assertIn('--enable-system-power-actions',window.power_notice.text())
        self.assertIn('Fase 9',window.power_notice.text())
        self.assertFalse(window.admin_actions['reboot'].isEnabled())
        self.assertFalse(window.admin_actions['poweroff'].isEnabled())
        self.power.execute.assert_not_called()

    def test_preference_reset_rejects_changed_snapshot_after_confirmation(self):
        self.login_keypad();self.w.navigate('settings')
        self.w.appearance.setCurrentIndex(1);self.w.timeout_selector.setCurrentIndex(2)
        def change_settings(dialog):
            self.admin.settings.update_standby_timeout(60)
        self.confirm_then_keypad(change_settings)
        self.w.admin_actions['preferences'].click()
        self.assertEqual(self.admin.settings.get().theme,'light')
        self.assertEqual(self.admin.settings.get().standby_timeout_seconds,60)
        self.assertIn('No se completó',self.w.settings_message.text())

    def production_restart(self,inspect):
        """Fresh-window production composition, same synthetic DB, no main event loop."""
        from face_recognition.presentation.gui.__main__ import main
        self.w.close();self.ui.finalize()
        expected_theme=self.admin.settings.get().theme
        windows=[];applied=[];reads=[];writes=[]
        original_apply=KioskWindow.apply_theme
        original_get=SQLiteDeviceSettingsRepository.get
        original_theme=SQLiteDeviceSettingsRepository.update_theme
        original_timeout=SQLiteDeviceSettingsRepository.update_standby_timeout
        def make_window(*args,**kwargs):
            result=KioskWindow(*args,**kwargs);windows.append(result);return result
        def apply(window,name):
            applied.append(name);return original_apply(window,name)
        def get(repo):
            reads.append(True);return original_get(repo)
        def theme(repo,value):
            writes.append(('theme',value));return original_theme(repo,value)
        def timeout(repo,value):
            writes.append(('timeout',value));return original_timeout(repo,value)
        def run():
            window=windows[-1]
            try:
                self.assertIs(window.lifecycle_state,LifecycleState.STANDBY)
                self.assertEqual(len(reads),1)
                self.assertEqual(writes,[])
                self.assertEqual(applied,[expected_theme])
                window.administration.service.clock=lambda:self.now
                inspect(window,applied,writes)
            finally:window.close()
            return 0
        with patch('PyQt5.QtWidgets.QApplication',return_value=self.app), \
                patch('face_recognition.presentation.gui.window.KioskWindow',side_effect=make_window), \
                patch.object(KioskWindow,'apply_theme',new=apply), \
                patch.object(SQLiteDeviceSettingsRepository,'get',new=get), \
                patch.object(SQLiteDeviceSettingsRepository,'update_theme',new=theme), \
                patch.object(SQLiteDeviceSettingsRepository,'update_standby_timeout',new=timeout), \
                patch.object(QApplication,'exec_',side_effect=run):
            self.assertEqual(main(['--windowed','--attendance-db',str(self.db.path)]),0)

    def login_window_keypad(self,window):
        window.open_admin()
        for digit in TEST_PIN:self.dialog_button(window.screens['pin'],digit).click()
        self.dialog_button(window.screens['pin'],'Confirmar').click()
        self.wait_for(lambda:window.administration.job is None)
        self.assertIs(window.lifecycle_state,LifecycleState.ADMIN)

    def assert_theme_appearance(self,window,name):
        """Check effective palettes AND rendered backgrounds, not theme metadata."""
        colors=THEMES[name]
        def palette(widget,role,token):
            widget.ensurePolished()
            self.assertEqual(widget.palette().color(role).name(),colors[token].lower(),type(widget).__name__)
        for page_name in ('dashboard','sessions','system','records','settings','export','sessions'):
            window.navigate(page_name);self.app.processEvents()
            if window.exports is not None:
                self.wait_for(lambda:not window.exports.busy)
            page=window.screens[page_name]
            palette(page,QPalette.Window,'background')
            self.assertEqual(page.grab().toImage().pixelColor(page.width()-3,page.height()-3).name(),colors['background'].lower())
            for scroll in page.findChildren(QScrollArea):
                for widget in (scroll.viewport(),scroll.widget()):palette(widget,QPalette.Window,'background')
                image=scroll.widget().grab().toImage()
                self.assertEqual(image.pixelColor(image.width()-4,image.height()-4).name(),colors['background'].lower())
            for tabs in page.findChildren(QTabWidget):
                palette(tabs.tabBar(),QPalette.Window,'background')
                for index in range(tabs.count()):palette(tabs.widget(index),QPalette.Window,'background')
            for table in page.findChildren(QTableWidget):
                palette(table,QPalette.Base,'card');palette(table.viewport(),QPalette.Base,'card')
                table.setRowCount(1);table.setItem(0,0,QTableWidgetItem('p001'))
                image=table.viewport().grab().toImage()
                self.assertEqual(image.pixelColor(image.width()-4,image.height()-4).name(),colors['card'].lower())
                self.assertEqual(table.horizontalHeader().grab().toImage().pixelColor(3,3).name(),colors['sidebar'].lower())
            for combo in page.findChildren(QComboBox):
                palette(combo,QPalette.Base,'card')
                combo.showPopup();self.app.processEvents()
                palette(combo.view(),QPalette.Base,'background');combo.hidePopup()
            for calendar in page.findChildren(QDateEdit):palette(calendar.calendarWidget(),QPalette.Window,'background')
            for frame in page.findChildren(QFrame):
                if frame.objectName()=='card':palette(frame,QPalette.Window,'card')
            for text in page.findChildren(QLabel):
                if text.objectName() in ('title','brand',''):
                    palette(text,QPalette.WindowText,'text')
            dynamic=QPushButton('Prueba',page);palette(dynamic,QPalette.Button,'card');dynamic.deleteLater()
        for page_name in ('standby','attendance','pin'):
            page=window.screens[page_name]
            palette(page,QPalette.Window,'background')
            image=page.grab().toImage()
            self.assertEqual(image.pixelColor(image.width()-3,image.height()-3).name(),colors['background'].lower())
        for widget in window.findChildren(QFrame):
            if widget.objectName() in ('sidebar','status'):palette(widget,QPalette.Window,'sidebar')
        def pin(dialog):
            field=dialog.findChildren(QLineEdit)[0]
            palette(field,QPalette.Base,'card')
            self.assertEqual(field.grab().toImage().pixelColor(field.width()//2,field.height()//2).name(),colors['card'].lower())
            self.dialog_button(dialog,'Cancelar').click()
        self.schedule_dialog(pin);self.assertIsNone(window.administration.reauthenticate('theme-visual-only'))
        def confirmation(dialog):
            palette(dialog,QPalette.Window,'background')
            palette(self.dialog_button(dialog,'Cancelar'),QPalette.Button,'card')
            self.dialog_button(dialog,'Cancelar').click()
        self.schedule_dialog(confirmation)
        self.assertFalse(window.administration.confirmation('Prueba de tema','Solo presentación','Continuar'))

    def test_startup_theme_light_effective_nested_appearance(self):
        self.admin.settings.update_theme('light')
        def check(window,applied,writes):
            self.assertEqual(window.appearance.currentIndex(),1)
            self.assertEqual(window.screens['standby'].palette().color(QPalette.Window).name(),THEMES['light']['background'].lower())
            window.navigate('attendance');self.app.processEvents()
            self.assertEqual(window.screens['attendance'].palette().color(QPalette.Window).name(),THEMES['light']['background'].lower())
            self.login_window_keypad(window);self.assert_theme_appearance(window,'light')
            self.assertEqual(applied,['light']);self.assertEqual(writes,[])
        self.production_restart(check)

    def test_startup_theme_dark_effective_nested_appearance(self):
        self.admin.settings.update_theme('dark')
        def check(window,applied,writes):
            self.login_window_keypad(window);self.assert_theme_appearance(window,'dark')
            self.assertEqual(applied,['dark']);self.assertEqual(writes,[])
        self.production_restart(check)

    def test_startup_theme_runtime_switching_keeps_hidden_pages_consistent(self):
        def check(window,applied,writes):
            self.login_window_keypad(window)
            for name in ('light','dark','light','dark'):
                window.appearance.setCurrentIndex(1 if name=='light' else 0)
                self.assert_theme_appearance(window,name)
        self.production_restart(check)

    def test_startup_theme_failed_preference_write_restores_effective_appearance(self):
        self.admin.settings.update_theme('light')
        def check(window,applied,writes):
            self.login_window_keypad(window)
            with patch.object(window.administration.service,'update_theme',side_effect=AttendancePersistenceError('Synthetic failure')):
                window.appearance.setCurrentIndex(0)
            self.assertEqual(window.appearance.currentIndex(),1)
            self.assertEqual(self.admin.settings.get().theme,'light')
            self.assert_theme_appearance(window,'light')
        self.production_restart(check)

    def test_startup_theme_authorized_preference_reset_and_restart_are_visually_dark(self):
        self.admin.settings.update_theme('light');self.admin.settings.update_standby_timeout(120)
        def reset(window,applied,writes):
            self.login_window_keypad(window);window.navigate('settings')
            self.confirm_then_keypad(lambda dialog:None)
            window.admin_actions['preferences'].click()
            self.assertEqual(window.standby_timeout_seconds,30)
            self.assertEqual(window.appearance.currentIndex(),0)
            self.assert_theme_appearance(window,'dark')
        self.production_restart(reset)
        def restored(window,applied,writes):
            self.assertEqual(window.standby_timeout_seconds,30)
            self.login_window_keypad(window);self.assert_theme_appearance(window,'dark')
            self.assertEqual(applied,['dark']);self.assertEqual(writes,[])
        self.production_restart(restored)

    def test_startup_theme_settings_read_failure_has_safe_single_dark_fallback(self):
        from face_recognition.presentation.gui.__main__ import main
        self.w.close();self.ui.finalize();windows=[]
        def window(*args,**kwargs):
            result=KioskWindow(*args,**kwargs);windows.append(result);return result
        def inspect():
            current=windows[-1]
            try:
                self.assertEqual(current.theme,'dark')
                self.assertIn('No fue posible cargar',current.settings_message.text())
                self.assertIs(current.lifecycle_state,LifecycleState.STANDBY)
            finally:current.close()
            return 0
        with patch('PyQt5.QtWidgets.QApplication',return_value=self.app), \
                patch('face_recognition.presentation.gui.window.KioskWindow',side_effect=window), \
                patch.object(SQLiteDeviceSettingsRepository,'get',side_effect=AttendancePersistenceError('Synthetic read failure')) as read, \
                patch.object(QApplication,'exec_',side_effect=inspect):
            self.assertEqual(main(['--windowed','--attendance-db',str(self.db.path)]),0)
            read.assert_called_once()
