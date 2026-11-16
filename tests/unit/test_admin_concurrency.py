"""Deterministic optimistic-authentication races; real SQLite, test-only derivation.

Production PBKDF2 concurrency is separately retained in test_administration.
No test relies on a derivation duration or a larger database busy timeout.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, local
import unittest
from unittest.mock import patch

from face_recognition.application.admin_service import AdminService
from face_recognition.domain.models.administration import AuthOutcome as Outcome
from face_recognition.domain.models.attendance import AttendancePersistenceError
from face_recognition.infrastructure.persistence import admin_credentials as credentials
from face_recognition.infrastructure.persistence.attendance_database import AttendanceDatabase, read_timestamp
from face_recognition.infrastructure.persistence.attendance_repositories import SQLiteDeviceSettingsRepository, SQLiteAttendanceSessionRepository, SQLiteParticipantRepository, SQLiteAttendanceRecordRepository
from tests.admin_fakes import TEST_PIN, fast_test_derive, use_fast_credentials

NEW_PIN = '092847'
OTHER_PIN = '817263'
NOW = datetime(2026, 11, 16, tzinfo=timezone.utc)


class CredentialConcurrencyTests(unittest.TestCase):
    def setUp(self):
        use_fast_credentials(self)
        self.temp=TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.db=AttendanceDatabase(Path(self.temp.name)/'synthetic.sqlite3',busy_timeout_ms=150)
        self.db.initialize()
        self.repo=credentials.SQLiteAdminCredentialRepository(self.db)
        self.now=NOW
        self.clock=lambda:self.now
        self.repo.setup(TEST_PIN,TEST_PIN,self.now)

    def row(self):
        with self.db.transaction() as c:return dict(c.execute('SELECT * FROM admin_credentials').fetchone())

    @contextmanager
    def pause_derivation(self, pin, *, occurrence=1):
        entered,release=Event(),Event()
        count=0
        def derive(value,*args):
            nonlocal count
            if value==pin:
                count+=1
                if count==occurrence:
                    entered.set()
                    if not release.wait(10):raise AssertionError('Test did not release derivation')
            return fast_test_derive(value,*args)
        with patch.object(credentials,'derive',side_effect=derive):
            try:yield entered,release
            finally:release.set()

    def test_derivation_has_no_open_transaction_and_allows_unrelated_writes(self):
        active=local(); original=self.db.transaction
        @contextmanager
        def observed(*args,**kwargs):
            with original(*args,**kwargs) as c:
                active.depth=getattr(active,'depth',0)+1
                try:yield c
                finally:active.depth-=1
        def derive(*args):
            self.assertEqual(getattr(active,'depth',0),0)
            SQLiteDeviceSettingsRepository(self.db).update_theme('light')
            session=SQLiteAttendanceSessionRepository(self.db).open('Unrelated',self.now,self.now.date())
            SQLiteAttendanceSessionRepository(self.db).close(session.id,self.now)
            return fast_test_derive(*args)
        with patch.object(self.db,'transaction',observed),patch.object(credentials,'derive',side_effect=derive):
            self.assertIs(self.repo.verify(TEST_PIN,self.clock)[0].outcome,Outcome.AUTHENTICATED)
        self.assertEqual(SQLiteDeviceSettingsRepository(self.db).get().theme,'light')

    def test_other_connections_can_commit_while_hash_is_paused(self):
        with ThreadPoolExecutor(max_workers=1) as pool,self.pause_derivation(TEST_PIN) as (entered,release):
            result=pool.submit(self.repo.verify,TEST_PIN,self.clock)
            try:
                self.assertTrue(entered.wait(5))
                SQLiteDeviceSettingsRepository(AttendanceDatabase(self.db.path,busy_timeout_ms=150)).update_standby_timeout(120)
                sessions=SQLiteAttendanceSessionRepository(AttendanceDatabase(self.db.path,busy_timeout_ms=150))
                session=sessions.open('Unrelated',self.now,self.now.date())
                SQLiteParticipantRepository(self.db).add('p001',self.now)
                record,created=SQLiteAttendanceRecordRepository(self.db).add('p001',session.id,self.now)
                self.assertTrue(created);self.assertEqual(record.attendance_session_id,session.id)
                sessions.close(session.id,self.now)
            finally:release.set()
            self.assertIs(result.result(timeout=5)[0].outcome,Outcome.AUTHENTICATED)

    def test_replaced_credential_cannot_authenticate_or_receive_old_failed_attempt(self):
        for submitted in (TEST_PIN,OTHER_PIN):
            with self.subTest(correct=submitted==TEST_PIN):
                # Restore a credential with a fresh salt, preserving actual repository semantics.
                if submitted==OTHER_PIN:self.repo.change(NEW_PIN,TEST_PIN,TEST_PIN,self.clock)
                with ThreadPoolExecutor(max_workers=1) as pool,self.pause_derivation(submitted) as (entered,release):
                    result=pool.submit(self.repo.verify,submitted,self.clock)
                    try:
                        self.assertTrue(entered.wait(5))
                        self.assertIs(self.repo.change(TEST_PIN,NEW_PIN,NEW_PIN,self.clock).outcome,Outcome.AUTHENTICATED)
                        newer=self.row()
                    finally:release.set()
                    self.assertIs(result.result(timeout=5)[0].outcome,Outcome.FAILED)
                self.assertEqual(self.row(),newer)

    def test_concurrent_pin_change_cannot_overwrite_replacement(self):
        original=self.repo.stamp()
        with ThreadPoolExecutor(max_workers=1) as pool,self.pause_derivation(NEW_PIN) as (entered,release):
            result=pool.submit(self.repo.change,TEST_PIN,NEW_PIN,NEW_PIN,self.clock)
            try:
                self.assertTrue(entered.wait(5))
                self.assertIs(self.repo.change(TEST_PIN,OTHER_PIN,OTHER_PIN,self.clock).outcome,Outcome.AUTHENTICATED)
                newer=self.row()
            finally:release.set()
            self.assertIs(result.result(timeout=5).outcome,Outcome.FAILED)
        self.assertNotEqual(self.repo.stamp(),original);self.assertEqual(self.row(),newer)
        self.assertIs(self.repo.verify(OTHER_PIN,self.clock)[0].outcome,Outcome.AUTHENTICATED)

    def test_success_and_failure_account_in_committed_order(self):
        for held,other,final in ((OTHER_PIN,TEST_PIN,1),(TEST_PIN,OTHER_PIN,0)):
            with self.db.transaction(write=True) as c:c.execute('UPDATE admin_credentials SET failed_attempts=0,locked_until=NULL')
            with ThreadPoolExecutor(max_workers=1) as pool,self.pause_derivation(held) as (entered,release):
                result=pool.submit(self.repo.verify,held,self.clock)
                try:
                    self.assertTrue(entered.wait(5))
                    self.repo.verify(other,self.clock)
                finally:release.set()
                result.result(timeout=5)
            self.assertEqual(self.row()['failed_attempts'],final)

    def test_inflight_verification_cannot_bypass_or_extend_new_lockout(self):
        for submitted in (TEST_PIN,NEW_PIN):
            with self.db.transaction(write=True) as c:c.execute('UPDATE admin_credentials SET failed_attempts=0,locked_until=NULL')
            with ThreadPoolExecutor(max_workers=1) as pool,self.pause_derivation(submitted) as (entered,release):
                result=pool.submit(self.repo.verify,submitted,self.clock)
                try:
                    self.assertTrue(entered.wait(5))
                    for _ in range(5):self.repo.verify(OTHER_PIN,self.clock)
                    locked=self.row()
                finally:release.set()
                self.assertIs(result.result(timeout=5)[0].outcome,Outcome.LOCKED)
            self.assertEqual(self.row(),locked)
            with patch.object(credentials,'derive',side_effect=AssertionError('Locked attempts must skip crypto')):
                self.assertIs(self.repo.verify(TEST_PIN,self.clock)[0].outcome,Outcome.LOCKED)

    def test_lockout_activated_during_pin_change_prevents_replacement(self):
        before=self.repo.stamp()
        with ThreadPoolExecutor(max_workers=1) as pool,self.pause_derivation(NEW_PIN) as (entered,release):
            result=pool.submit(self.repo.change,TEST_PIN,NEW_PIN,NEW_PIN,self.clock)
            try:
                self.assertTrue(entered.wait(5))
                for _ in range(5):self.repo.verify(OTHER_PIN,self.clock)
                locked=self.row()
            finally:release.set()
            self.assertIs(result.result(timeout=5).outcome,Outcome.LOCKED)
        self.assertEqual(self.repo.stamp(),before);self.assertEqual(self.row(),locked)

    def test_pin_change_does_not_reset_counter_before_commit_or_hold_transaction(self):
        self.repo.verify(OTHER_PIN,self.clock)
        with ThreadPoolExecutor(max_workers=1) as pool,self.pause_derivation(NEW_PIN) as (entered,release):
            result=pool.submit(self.repo.change,TEST_PIN,NEW_PIN,NEW_PIN,self.clock)
            try:
                self.assertTrue(entered.wait(5))
                self.assertEqual(self.row()['failed_attempts'],1)
                SQLiteDeviceSettingsRepository(self.db).update_theme('light')
                self.repo.verify(OTHER_PIN,self.clock)
                self.assertEqual(self.row()['failed_attempts'],2)
            finally:release.set()
            self.assertIs(result.result(timeout=5).outcome,Outcome.AUTHENTICATED)
        self.assertEqual(self.row()['failed_attempts'],0)
        self.assertIs(self.repo.verify(NEW_PIN,self.clock)[0].outcome,Outcome.AUTHENTICATED)

    def test_lock_duration_starts_after_derivation_and_write_lock_acquisition(self):
        with self.db.transaction(write=True) as c:c.execute('UPDATE admin_credentials SET failed_attempts=4')
        def derive(*args):
            self.now+=timedelta(seconds=20)
            return fast_test_derive(*args)
        original=self.db.transaction
        @contextmanager
        def contended(*args,**kwargs):
            with original(*args,**kwargs) as c:
                if kwargs.get('write'):self.now+=timedelta(seconds=7)
                yield c
        with patch.object(credentials,'derive',side_effect=derive),patch.object(self.db,'transaction',contended):
            result,_=self.repo.verify(OTHER_PIN,self.clock)
        self.assertEqual(result.retry_after,60)
        self.assertEqual(read_timestamp(self.row()['locked_until']),NOW+timedelta(seconds=87))

    def test_clock_rollback_during_derivation_fails_closed_without_changes(self):
        before=self.row()
        def derive(*args):
            self.now-=timedelta(seconds=1)
            return fast_test_derive(*args)
        with patch.object(credentials,'derive',side_effect=derive):
            self.assertIs(self.repo.verify(TEST_PIN,self.clock)[0].outcome,Outcome.LOCKED)
        self.assertEqual(self.row(),before)

    def test_write_failure_does_not_authenticate_or_partially_replace_credentials(self):
        admin=AdminService(self.repo,clock=self.clock)
        original=self.db.transaction; before=self.row()
        @contextmanager
        def fail(*args,**kwargs):
            if kwargs.get('write'):raise AttendancePersistenceError('Synthetic write failure')
            with original(*args,**kwargs) as c:yield c
        with patch.object(self.db,'transaction',fail):
            self.assertIs(admin.authenticate(TEST_PIN).outcome,Outcome.FAILED)
            with self.assertRaises(PermissionError):admin.require_authenticated()
            with self.assertRaises(AttendancePersistenceError):self.repo.change(TEST_PIN,NEW_PIN,NEW_PIN,self.clock)
        self.assertEqual(self.row(),before)

    def test_failure_after_update_rolls_back_counter_and_credential_replacement(self):
        self.repo.verify(OTHER_PIN,self.clock)
        before=self.row(); original=self.db.transaction
        @contextmanager
        def fail_after_update(*args,**kwargs):
            with original(*args,**kwargs) as c:
                yield c
                if kwargs.get('write'):raise AttendancePersistenceError('Synthetic commit failure')
        admin=AdminService(self.repo,clock=self.clock)
        with patch.object(self.db,'transaction',fail_after_update):
            self.assertIs(admin.authenticate(TEST_PIN).outcome,Outcome.FAILED)
            with self.assertRaises(PermissionError):admin.require_authenticated()
            with self.assertRaises(AttendancePersistenceError):self.repo.change(TEST_PIN,NEW_PIN,NEW_PIN,self.clock)
        self.assertEqual(self.row(),before)
