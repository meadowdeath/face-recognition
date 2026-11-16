"""Administration security and migrations, using fictional credentials and disposable storage."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, redirect_stdout
from datetime import datetime, timedelta, timezone, date
from io import StringIO
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from face_recognition.application.admin_service import AdminService
from face_recognition.application.attendance_service import AttendanceService, RegistrationOutcome
from face_recognition.domain.models.administration import AuthOutcome as Outcome, MaintenanceOperation as Op, validate_pin
from face_recognition.domain.models.attendance import AttendancePersistenceError
from face_recognition.infrastructure.persistence.admin_credentials import SQLiteAdminCredentialRepository
from face_recognition.infrastructure.persistence.attendance_database import AttendanceDatabase, timestamp
from face_recognition.infrastructure.persistence.attendance_repositories import SQLiteAdministrationRepository, SQLiteParticipantRepository, SQLiteAttendanceSessionRepository, SQLiteAttendanceRecordRepository, SQLiteDeviceSettingsRepository
from tests.admin_fakes import TEST_PIN, grant, use_fast_credentials
from face_recognition.infrastructure.persistence import admin_credentials as credential_module

from tests.attendance_v3_schema import create_v3
from tests.attendance_v2_schema import create_v2
from tests.attendance_v1_schema import create_v1

PRODUCTION_DERIVE = credential_module.derive
NOW = datetime(2026, 11, 16, tzinfo=timezone.utc)
NEW_PIN = '092847'


class AdministrationTests(unittest.TestCase):
    REAL_CRYPTO_TESTS = frozenset((
        'test_hash_salt_constant_time_and_plaintext_not_stored',
        'test_change_pin_verifies_current_new_salt_logout_old_rejection',
        'test_concurrent_failed_attempts_are_atomic',
    ))

    def setUp(self):
        if self._testMethodName in self.REAL_CRYPTO_TESTS:
            self.assertIs(credential_module.derive, PRODUCTION_DERIVE)
        else:
            use_fast_credentials(self)
        self.temp = TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'synthetic.sqlite3'
        self.db = AttendanceDatabase(self.path); self.db.initialize()
        self.now, self.elapsed = NOW, 10.
        self.repo = SQLiteAdminCredentialRepository(self.db)
        self.settings = SQLiteDeviceSettingsRepository(self.db)
        self.people = SQLiteParticipantRepository(self.db)
        self.sessions = SQLiteAttendanceSessionRepository(self.db)
        self.records = SQLiteAttendanceRecordRepository(self.db)
        self.admin = AdminService(self.repo, clock=lambda: self.now, timer=lambda: self.elapsed,
            administration=SQLiteAdministrationRepository(self.db), settings=self.settings, participants=self.people)

    def setup_admin(self):
        self.repo.setup(TEST_PIN, TEST_PIN, self.now)
        self.assertIs(self.admin.authenticate(TEST_PIN).outcome, Outcome.AUTHENTICATED)

    def test_empty_credential_schema_idempotent_and_setup_no_default(self):
        self.assertFalse(self.admin.configured())
        self.assertIs(self.admin.authenticate('123456').outcome, Outcome.SETUP_REQUIRED)
        self.db.initialize()
        with self.db.transaction() as c:
            self.assertEqual(c.execute('PRAGMA user_version').fetchone()[0], 4)
            self.assertEqual(c.execute('SELECT count(*) FROM admin_credentials').fetchone()[0], 0)
        self.setup_admin()
        before = self.repo.stamp()
        self.db.initialize(); self.assertEqual(before, self.repo.stamp())
        with self.assertRaises(ValueError): self.repo.setup(NEW_PIN, NEW_PIN, self.now)
        self.assertEqual(before, self.repo.stamp())

    def test_pin_validation_ascii_leading_zeroes_confirmation_and_trivial_values(self):
        self.assertEqual(validate_pin(TEST_PIN, TEST_PIN, new=True), TEST_PIN)
        for bad in ('123456','000000','111111','999999','654321','012345','', '12345','1234567','１２３４５６','12a456',None):
            with self.subTest(value=bad), self.assertRaises(ValueError): self.repo.setup(bad,bad,self.now)
        with self.assertRaises(ValueError): self.repo.setup(TEST_PIN, NEW_PIN, self.now)
        self.assertFalse(self.admin.configured())

    def test_hash_salt_constant_time_and_plaintext_not_stored(self):
        self.setup_admin()
        with self.db.transaction() as c: row = c.execute('SELECT * FROM admin_credentials').fetchone()
        self.assertEqual(len(row['pin_hash']),32); self.assertGreaterEqual(len(row['pin_salt']),16)
        self.assertEqual(row['iterations'],600000)
        self.assertNotIn(TEST_PIN.encode(), self.path.read_bytes())
        reopened = SQLiteAdminCredentialRepository(AttendanceDatabase(self.path))
        self.assertIs(reopened.verify(TEST_PIN, lambda:self.now)[0].outcome, Outcome.AUTHENTICATED)
        with patch('face_recognition.infrastructure.persistence.admin_credentials.compare_digest', wraps=__import__('hmac').compare_digest) as compare:
            self.assertIs(self.admin.authenticate(TEST_PIN).outcome, Outcome.AUTHENTICATED)
            compare.assert_called_once()
        other = AttendanceDatabase(Path(self.temp.name)/'other.sqlite3'); other.initialize()
        SQLiteAdminCredentialRepository(other).setup(TEST_PIN, TEST_PIN, self.now)
        self.assertNotEqual(self.repo.stamp()[1], SQLiteAdminCredentialRepository(other).stamp()[1])

    def test_five_attempt_lockout_persists_and_skips_hash_until_expired(self):
        self.setup_admin()
        for _ in range(4): self.assertIs(self.admin.authenticate(NEW_PIN).outcome,Outcome.INCORRECT)
        self.assertEqual(self.admin.authenticate(NEW_PIN).retry_after,60)
        restarted = AdminService(SQLiteAdminCredentialRepository(AttendanceDatabase(self.path)), clock=lambda:self.now)
        with patch('face_recognition.infrastructure.persistence.admin_credentials.derive', side_effect=AssertionError('No derivation while locked')):
            self.assertIs(restarted.authenticate(TEST_PIN).outcome,Outcome.LOCKED)
        self.now += timedelta(seconds=61)
        self.assertIs(restarted.authenticate(TEST_PIN).outcome,Outcome.AUTHENTICATED)
        with self.db.transaction() as c:
            row=c.execute('SELECT failed_attempts,locked_until FROM admin_credentials').fetchone()
            self.assertEqual(tuple(row),(0,None))
        self.now = NOW-timedelta(seconds=1)
        self.assertIs(restarted.authenticate(TEST_PIN).outcome, Outcome.LOCKED)

    def test_concurrent_failed_attempts_are_atomic(self):
        self.setup_admin()
        from threading import Barrier
        ready = Barrier(5)
        def concurrent_real_derive(*args):
            ready.wait(timeout=10)
            return PRODUCTION_DERIVE(*args)
        with patch.object(credential_module, 'derive', side_effect=concurrent_real_derive), ThreadPoolExecutor(max_workers=5) as pool:
            results=list(pool.map(lambda _: self.repo.verify(NEW_PIN,lambda:self.now)[0],range(5)))
        self.assertEqual(sum(r.outcome is Outcome.INCORRECT for r in results),4)
        self.assertEqual(sum(r.outcome is Outcome.LOCKED for r in results),1)
        with self.db.transaction() as c: self.assertEqual(c.execute('SELECT failed_attempts FROM admin_credentials').fetchone()[0],5)

    def test_change_pin_verifies_current_new_salt_logout_old_rejection(self):
        self.setup_admin(); before=self.repo.stamp()
        self.assertIs(self.admin.change_pin(NEW_PIN,NEW_PIN,NEW_PIN).outcome,Outcome.INVALID)
        self.assertEqual(self.repo.stamp(),before)
        self.assertIs(self.admin.change_pin(NEW_PIN,'817263','817263').outcome,Outcome.INCORRECT)
        result=self.admin.change_pin(TEST_PIN,NEW_PIN,NEW_PIN)
        self.assertIs(result.outcome,Outcome.AUTHENTICATED)
        self.assertNotEqual(self.repo.stamp()[1],before[1])
        with self.assertRaises(PermissionError): self.admin.require_authenticated()
        self.assertIs(self.admin.authenticate(TEST_PIN).outcome,Outcome.INCORRECT)
        self.assertIs(self.admin.authenticate(NEW_PIN).outcome,Outcome.AUTHENTICATED)

    def test_authorizations_expire_are_single_use_and_bound_to_scope(self):
        self.setup_admin()
        def permit(): return self.admin.authorize('scope',TEST_PIN)[1]
        token=permit(); token.require('scope')
        with self.assertRaises(PermissionError): token.require('scope')
        token=permit()
        with self.assertRaises(PermissionError): token.require('other')
        with self.assertRaises(PermissionError): token.require('scope')
        token=permit(); self.elapsed+=61
        with self.assertRaises(PermissionError): token.require('scope')
        token=permit(); self.admin.invalidate_authorizations()
        with self.assertRaises(PermissionError): token.require('scope')
        token=permit(); self.admin.logout()
        with self.assertRaises(PermissionError): token.require('scope')
        with self.assertRaises(PermissionError): AdminService(self.repo).require_authenticated()

    def test_pin_change_invalidates_pending_grant_and_reauthentication_is_limited(self):
        self.setup_admin()
        token=self.admin.authorize('scope',TEST_PIN)[1]
        self.admin.change_pin(TEST_PIN,NEW_PIN,NEW_PIN)
        with self.assertRaises(PermissionError): token.require('scope')
        self.admin.authenticate(NEW_PIN)
        for _ in range(5): result, token=self.admin.authorize('scope',TEST_PIN)
        self.assertIs(result.outcome,Outcome.LOCKED); self.assertIsNone(token)

    def seed(self):
        self.setup_admin()
        for pid in ('p001','p002'): self.people.add(pid,self.now)
        session=self.sessions.open('Synthetic',self.now,date(2026,11,16))
        self.records.add('p001',session.id,self.now)
        self.sessions.close(session.id,self.now)
        return session

    def test_global_scopes_preserve_credentials_preferences_and_sequences(self):
        session=self.seed(); self.settings.update_theme('light')
        before=self.repo.stamp()
        with self.db.transaction() as c: sequences=dict(c.execute('SELECT name,seq FROM sqlite_sequence'))
        preview=self.admin.prepare_maintenance(Op.HISTORY)
        self.assertEqual((preview.sessions,preview.records,preview.participants),(1,1,2))
        self.admin.execute_maintenance(preview,grant(self.admin,preview))
        self.assertEqual(len(self.people.list()),2); self.assertFalse(self.sessions.list())
        preview=self.admin.prepare_maintenance(Op.ADMIN_DATA)
        self.admin.execute_maintenance(preview,grant(self.admin,preview))
        self.assertFalse(self.people.list()); self.assertEqual(self.repo.stamp(),before)
        self.assertEqual(self.settings.get().theme,'light')
        with self.db.transaction() as c: self.assertEqual(dict(c.execute('SELECT name,seq FROM sqlite_sequence')),sequences)
        recreated=self.sessions.open('Synthetic',self.now,date(2026,11,16)); self.assertGreater(recreated.id,session.id)
        self.assertIs(AttendanceService(self.people,self.sessions,self.records).register('p001',recreated.id).outcome,RegistrationOutcome.UNREGISTERED)

    def test_changed_global_snapshot_and_equal_count_replacement_abort(self):
        session=self.seed()
        for change in ('participant','record'):
            preview=self.admin.prepare_maintenance(Op.HISTORY)
            if change=='participant': self.people.add('p003',self.now)
            else:
                with self.db.transaction(write=True) as c:
                    c.execute('DELETE FROM attendance_records')
                    c.execute('INSERT INTO attendance_records(person_id,attendance_session_id,registered_at) VALUES (?,?,?)',('p001',session.id,timestamp(self.now)))
            token=grant(self.admin,preview)
            with self.assertRaisesRegex(ValueError,'changed'): self.admin.execute_maintenance(preview,token)
            with self.assertRaises(PermissionError): token.require(preview.fingerprint)
        self.assertEqual(self.records.count_for_session(session.id),1)

    def test_active_session_blocks_history_reset_but_preferences_preserve_it(self):
        session=self.seed(); active=self.sessions.resume(session.id).session
        for op in (Op.HISTORY,Op.ADMIN_DATA):
            with self.assertRaises(ValueError): self.admin.prepare_maintenance(op)
        self.settings.update_theme('light'); self.settings.update_standby_timeout(120)
        preview=self.admin.prepare_maintenance(Op.PREFERENCES)
        self.admin.execute_maintenance(preview,grant(self.admin,preview))
        self.assertEqual((self.settings.get().theme,self.settings.get().standby_timeout_seconds),('dark',30))
        self.assertEqual(self.sessions.current(),active); self.assertEqual(self.records.count_for_session(session.id),1)

    def test_setup_cli_prompts_not_arguments_and_repeated_setup_no_prompt(self):
        from face_recognition.presentation.cli.admin import main
        args=['--attendance-db',str(self.path),'setup']
        with patch('face_recognition.presentation.cli.admin.getpass',side_effect=[TEST_PIN,TEST_PIN]) as prompt, redirect_stdout(StringIO()) as output:
            self.assertEqual(main(args),0)
            self.assertEqual(prompt.call_count,2)
            self.assertNotIn(TEST_PIN,output.getvalue())
        with patch('face_recognition.presentation.cli.admin.getpass',side_effect=AssertionError('No overwrite')), self.assertRaises(SystemExit): main(args)

    def test_administrative_session_writes_fail_closed_without_login(self):
        service=AttendanceService(self.people,self.sessions,self.records,administrator=self.admin,clock=lambda:self.now)
        for operation in (lambda:service.create_session('Synthetic',self.now.date()),
                          lambda:service.open_session('Synthetic'),lambda:service.resume_session(1),
                          lambda:service.edit_session(1,'Synthetic',self.now.date()),lambda:service.close_session(1)):
            with self.assertRaises(PermissionError):operation()
        unsecured=AttendanceService(self.people,self.sessions,self.records)
        with self.assertRaises(PermissionError):unsecured.open_session('No bypass')

    def test_global_reset_rollback_preserves_every_table_when_participant_delete_fails(self):
        from contextlib import contextmanager
        session=self.seed(); preview=self.admin.prepare_maintenance(Op.ADMIN_DATA)
        token=grant(self.admin,preview); connection=self.db._connection
        @contextmanager
        def denied():
            with connection() as c:
                c.set_authorizer(lambda action,table,*args:sqlite3.SQLITE_DENY
                    if action==sqlite3.SQLITE_DELETE and table=='participants' else sqlite3.SQLITE_OK)
                yield c
        with patch.object(self.db,'_connection',denied):
            with self.assertRaises(AttendancePersistenceError):self.admin.execute_maintenance(preview,token)
        self.assertEqual(len(self.people.list()),2); self.assertIsNotNone(self.sessions.get(session.id))
        self.assertEqual(self.records.count_for_session(session.id),1)
        with self.assertRaises(PermissionError):token.require(preview.fingerprint)

    def test_pin_change_derivation_failure_rolls_back_credential_and_attempt_reset(self):
        self.setup_admin(); before=self.repo.stamp()
        self.repo.verify(NEW_PIN,lambda:self.now)
        from face_recognition.infrastructure.persistence.admin_credentials import derive as real_derive
        count=0
        def fail_second(*args):
            nonlocal count
            count+=1
            if count==2:raise ValueError('Synthetic derivation failure')
            return real_derive(*args)
        with patch('face_recognition.infrastructure.persistence.admin_credentials.derive',side_effect=fail_second):
            self.assertIs(self.admin.change_pin(TEST_PIN,NEW_PIN,NEW_PIN).outcome,Outcome.INVALID)
        self.assertEqual(self.repo.stamp(),before)
        with self.db.transaction() as c:self.assertEqual(c.execute('SELECT failed_attempts FROM admin_credentials').fetchone()[0],1)

    def test_bounded_lockout_escalation_and_credential_constraints(self):
        self.setup_admin()
        for failed,seconds in ((4,60),(9,120),(14,240),(19,240)):
            with self.db.transaction(write=True) as c:c.execute('UPDATE admin_credentials SET failed_attempts=?,locked_until=NULL WHERE id=1',(failed,))
            self.assertEqual(self.repo.verify(NEW_PIN,lambda:self.now)[0].retry_after,seconds)
            self.now+=timedelta(seconds=seconds+1)
        for column,value in (('pin_hash',b'bad'),('pin_salt',b'short'),('algorithm','sha256'),('iterations',1000),('failed_attempts',-1),('locked_until','invalid'),('id',2)):
            with self.subTest(column=column),self.assertRaises(ValueError):
                with self.db.transaction(write=True) as c:c.execute('UPDATE admin_credentials SET '+column+'=? WHERE id=1',(value,))

    def test_logout_during_pending_hash_cannot_establish_login(self):
        from threading import Event
        self.setup_admin(); entered,release=Event(),Event(); verify=self.repo.verify
        def delayed(*args):
            entered.set();release.wait(3);return verify(*args)
        with patch.object(self.repo,'verify',side_effect=delayed),ThreadPoolExecutor(max_workers=1) as pool:
            result=pool.submit(self.admin.authenticate,TEST_PIN)
            try:
                self.assertTrue(entered.wait(3));self.admin.logout()
            finally:release.set()
            self.assertIs(result.result(timeout=3).outcome,Outcome.FAILED)
        with self.assertRaises(PermissionError):self.admin.require_authenticated()


class SchemaFourTests(unittest.TestCase):
    def test_genuine_historical_migrations_preserve_all_data_and_sequence_high_water(self):
        with TemporaryDirectory() as temp:
            for version, create in ((1,create_v1),(2,create_v2),(3,create_v3)):
                path=Path(temp)/f'v{version}.sqlite3'; create(path)
                with closing(sqlite3.connect(path)) as c:
                    c.execute('PRAGMA foreign_keys=ON')
                    c.execute('INSERT INTO participants VALUES (?,?)',('p001',timestamp(NOW)))
                    columns='id,name,opened_at,closed_at'; values=[7,'Synthetic',timestamp(NOW),timestamp(NOW)]
                    if version>=2: columns+=',name_key,activation_revision'; values+=['synthetic',3]
                    if version>=3: columns+=',session_date'; values+=['2026-11-16']
                    c.execute('INSERT INTO attendance_sessions ('+columns+') VALUES ('+','.join('?' for _ in values)+')',values)
                    c.execute('INSERT INTO attendance_records VALUES (23,?,?,?)',('p001',7,timestamp(NOW)))
                    c.execute("UPDATE device_settings SET theme='light',standby_timeout_seconds=120")
                    if version==3:
                        c.execute("UPDATE sqlite_sequence SET seq=100 WHERE name='attendance_sessions'")
                        c.execute("UPDATE sqlite_sequence SET seq=200 WHERE name='attendance_records'")
                    c.commit()
                db=AttendanceDatabase(path); mode=path.stat().st_mode; db.initialize(); db.initialize()
                with db.transaction() as c:
                    self.assertEqual(c.execute('PRAGMA user_version').fetchone()[0],4)
                    self.assertFalse(c.execute('PRAGMA foreign_key_check').fetchall())
                    self.assertFalse(c.execute('SELECT * FROM admin_credentials').fetchall())
                    self.assertEqual(c.execute('SELECT opened_at FROM attendance_sessions').fetchone()[0],timestamp(NOW))
                    self.assertEqual(c.execute('SELECT id FROM attendance_records').fetchone()[0],23)
                    seq=dict(c.execute('SELECT name,seq FROM sqlite_sequence'))
                    self.assertEqual(seq,{'attendance_sessions':100 if version==3 else 7,'attendance_records':200 if version==3 else 23})
                self.assertEqual(SQLiteDeviceSettingsRepository(db).get().theme,'light')
                self.assertEqual(path.stat().st_mode,mode)

    def test_v3_migration_failure_rolls_back_every_object_and_version(self):
        with TemporaryDirectory() as temp:
            path=Path(temp)/'v3.sqlite3'; create_v3(path)
            with closing(sqlite3.connect(path)) as c: before='\n'.join(c.iterdump())
            with patch('face_recognition.infrastructure.persistence.attendance_database.ADMIN_SCHEMA','INVALID SQL'):
                with self.assertRaises(AttendancePersistenceError): AttendanceDatabase(path).initialize()
            with closing(sqlite3.connect(path)) as c: self.assertEqual('\n'.join(c.iterdump()),before)
