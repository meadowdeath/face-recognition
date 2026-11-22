"""Frozen v4 fixture for migration tests."""
from contextlib import closing
import sqlite3

from tests.attendance_v3_schema import create_v3


ADMIN_TABLE = """CREATE TABLE admin_credentials (
    id INTEGER PRIMARY KEY CHECK (id=1),
    pin_hash BLOB NOT NULL CHECK(typeof(pin_hash)='blob'
        AND length(pin_hash)=32),
    pin_salt BLOB NOT NULL CHECK(typeof(pin_salt)='blob'
        AND length(pin_salt)>=16),
    algorithm TEXT NOT NULL CHECK(algorithm='pbkdf2_sha256_v1'),
    iterations INTEGER NOT NULL CHECK(typeof(iterations)='integer'
        AND iterations>=600000),
    failed_attempts INTEGER NOT NULL DEFAULT 0 CHECK(typeof(failed_attempts)='integer'
        AND failed_attempts>=0),
    locked_until TEXT CHECK (locked_until IS NULL OR (length(locked_until) = 27
        AND instr(locked_until, char(0)) = 0
        AND locked_until GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z'
        AND substr(locked_until,1,4) >= '0001'
        AND substr(locked_until,12,2) BETWEEN '00'
        AND '23'
        AND substr(locked_until,15,2) BETWEEN '00'
        AND '59'
        AND substr(locked_until,18,2) BETWEEN '00'
        AND '59'
        AND date(substr(locked_until,1,10), '+0 days') IS NOT NULL
        AND date(substr(locked_until,1,10), '+0 days') = substr(locked_until,1,10)
        AND strftime('%Y-%m-%dT%H:%M:%S', substr(locked_until,1,19)) IS NOT NULL
        AND strftime('%Y-%m-%dT%H:%M:%S', substr(locked_until,1,19)) = substr(locked_until,1,19))),
    created_at TEXT NOT NULL CHECK (created_at IS NULL OR (length(created_at) = 27
        AND instr(created_at, char(0)) = 0
        AND created_at GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z'
        AND substr(created_at,1,4) >= '0001'
        AND substr(created_at,12,2) BETWEEN '00'
        AND '23'
        AND substr(created_at,15,2) BETWEEN '00'
        AND '59'
        AND substr(created_at,18,2) BETWEEN '00'
        AND '59'
        AND date(substr(created_at,1,10), '+0 days') IS NOT NULL
        AND date(substr(created_at,1,10), '+0 days') = substr(created_at,1,10)
        AND strftime('%Y-%m-%dT%H:%M:%S', substr(created_at,1,19)) IS NOT NULL
        AND strftime('%Y-%m-%dT%H:%M:%S', substr(created_at,1,19)) = substr(created_at,1,19))),
    updated_at TEXT NOT NULL CHECK (updated_at IS NULL OR (length(updated_at) = 27
        AND instr(updated_at, char(0)) = 0
        AND updated_at GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z'
        AND substr(updated_at,1,4) >= '0001'
        AND substr(updated_at,12,2) BETWEEN '00'
        AND '23'
        AND substr(updated_at,15,2) BETWEEN '00'
        AND '59'
        AND substr(updated_at,18,2) BETWEEN '00'
        AND '59'
        AND date(substr(updated_at,1,10), '+0 days') IS NOT NULL
        AND date(substr(updated_at,1,10), '+0 days') = substr(updated_at,1,10)
        AND strftime('%Y-%m-%dT%H:%M:%S', substr(updated_at,1,19)) IS NOT NULL
        AND strftime('%Y-%m-%dT%H:%M:%S', substr(updated_at,1,19)) = substr(updated_at,1,19))),
    CHECK(updated_at>=created_at))"""


def create_v4(path):
    create_v3(path)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(ADMIN_TABLE)
        connection.execute("PRAGMA user_version=4")
