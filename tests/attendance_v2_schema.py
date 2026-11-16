"""Genuine schema v2 snapshot, independent of subsequent production migrations."""
from contextlib import closing
import sqlite3

SCHEMA_V2 = ('CREATE TABLE participants (\n'
 '        person_id TEXT PRIMARY KEY NOT NULL CHECK (\n'
 "            length(person_id) >= 4 AND instr(person_id, char(0)) = 0 AND substr(person_id,1,1) = 'p'\n"
 "            AND substr(person_id,2) NOT GLOB '*[^0-9]*'),\n"
 '        created_at TEXT NOT NULL CHECK (created_at IS NULL OR (length(created_at) = 27 AND '
 'instr(created_at, char(0)) = 0 AND created_at GLOB '
 "'[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z' "
 "AND substr(created_at,1,4) >= '0001' AND substr(created_at,12,2) BETWEEN '00' AND '23' AND "
 "substr(created_at,15,2) BETWEEN '00' AND '59' AND substr(created_at,18,2) BETWEEN '00' AND '59' AND "
 "date(substr(created_at,1,10), '+0 days') IS NOT NULL AND date(substr(created_at,1,10), '+0 days') = "
 "substr(created_at,1,10) AND strftime('%Y-%m-%dT%H:%M:%S', substr(created_at,1,19)) IS NOT NULL AND "
 "strftime('%Y-%m-%dT%H:%M:%S', substr(created_at,1,19)) = substr(created_at,1,19))))",
 'CREATE TABLE attendance_sessions (\n'
 '        id INTEGER PRIMARY KEY,\n'
 '        name TEXT NOT NULL CHECK (length(trim(name, char(9,10,11,12,13,32))) > 0),\n'
 '        opened_at TEXT NOT NULL CHECK (opened_at IS NULL OR (length(opened_at) = 27 AND instr(opened_at, '
 'char(0)) = 0 AND opened_at GLOB '
 "'[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z' "
 "AND substr(opened_at,1,4) >= '0001' AND substr(opened_at,12,2) BETWEEN '00' AND '23' AND "
 "substr(opened_at,15,2) BETWEEN '00' AND '59' AND substr(opened_at,18,2) BETWEEN '00' AND '59' AND "
 "date(substr(opened_at,1,10), '+0 days') IS NOT NULL AND date(substr(opened_at,1,10), '+0 days') = "
 "substr(opened_at,1,10) AND strftime('%Y-%m-%dT%H:%M:%S', substr(opened_at,1,19)) IS NOT NULL AND "
 "strftime('%Y-%m-%dT%H:%M:%S', substr(opened_at,1,19)) = substr(opened_at,1,19))),\n"
 '        closed_at TEXT CHECK (closed_at IS NULL OR (length(closed_at) = 27 AND instr(closed_at, char(0)) = '
 '0 AND closed_at GLOB '
 "'[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z' "
 "AND substr(closed_at,1,4) >= '0001' AND substr(closed_at,12,2) BETWEEN '00' AND '23' AND "
 "substr(closed_at,15,2) BETWEEN '00' AND '59' AND substr(closed_at,18,2) BETWEEN '00' AND '59' AND "
 "date(substr(closed_at,1,10), '+0 days') IS NOT NULL AND date(substr(closed_at,1,10), '+0 days') = "
 "substr(closed_at,1,10) AND strftime('%Y-%m-%dT%H:%M:%S', substr(closed_at,1,19)) IS NOT NULL AND "
 "strftime('%Y-%m-%dT%H:%M:%S', substr(closed_at,1,19)) = substr(closed_at,1,19))),\n"
 "        name_key TEXT NOT NULL DEFAULT '',\n"
 "        activation_revision INTEGER NOT NULL DEFAULT 1 CHECK (typeof(activation_revision) = 'integer' AND "
 'activation_revision > 0),\n'
 '        CHECK (closed_at IS NULL OR closed_at >= opened_at))',
 'CREATE UNIQUE INDEX one_open_attendance_session ON attendance_sessions ((1)) WHERE closed_at IS NULL',
 'CREATE TABLE attendance_records (\n'
 '        id INTEGER PRIMARY KEY,\n'
 '        person_id TEXT NOT NULL REFERENCES participants(person_id) ON DELETE CASCADE,\n'
 '        attendance_session_id INTEGER NOT NULL REFERENCES attendance_sessions(id) ON DELETE RESTRICT,\n'
 '        registered_at TEXT NOT NULL CHECK (registered_at IS NULL OR (length(registered_at) = 27 AND '
 'instr(registered_at, char(0)) = 0 AND registered_at GLOB '
 "'[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z' "
 "AND substr(registered_at,1,4) >= '0001' AND substr(registered_at,12,2) BETWEEN '00' AND '23' AND "
 "substr(registered_at,15,2) BETWEEN '00' AND '59' AND substr(registered_at,18,2) BETWEEN '00' AND '59' AND "
 "date(substr(registered_at,1,10), '+0 days') IS NOT NULL AND date(substr(registered_at,1,10), '+0 days') = "
 "substr(registered_at,1,10) AND strftime('%Y-%m-%dT%H:%M:%S', substr(registered_at,1,19)) IS NOT NULL AND "
 "strftime('%Y-%m-%dT%H:%M:%S', substr(registered_at,1,19)) = substr(registered_at,1,19))),\n"
 '        UNIQUE (attendance_session_id, person_id))',
 'CREATE INDEX attendance_by_participant ON attendance_records (person_id, registered_at, id)',
 'CREATE INDEX attendance_by_timestamp ON attendance_records (registered_at, id)',
 'CREATE TABLE device_settings (\n'
 '        id INTEGER PRIMARY KEY CHECK (id = 1),\n'
 "        theme TEXT NOT NULL CHECK (theme IN ('dark', 'light')),\n"
 '        standby_timeout_seconds INTEGER NOT NULL CHECK (\n'
 "            typeof(standby_timeout_seconds) = 'integer' AND standby_timeout_seconds IN (30,60,120)))",
 'CREATE UNIQUE INDEX unique_session_name ON attendance_sessions (name_key)',
 'CREATE TRIGGER valid_session_name_key_insert\n'
 '        BEFORE INSERT ON attendance_sessions\n'
 "        WHEN typeof(NEW.name_key) != 'text' OR length(trim(NEW.name_key, char(9,10,11,12,13,32))) = 0\n"
 '             OR instr(NEW.name_key, char(0)) != 0\n'
 "        BEGIN SELECT RAISE(ABORT, 'Session name key must not be empty or contain NUL'); END",
 'CREATE TRIGGER valid_session_name_key_update\n'
 '        BEFORE UPDATE ON attendance_sessions\n'
 "        WHEN typeof(NEW.name_key) != 'text' OR length(trim(NEW.name_key, char(9,10,11,12,13,32))) = 0\n"
 '             OR instr(NEW.name_key, char(0)) != 0\n'
 "        BEGIN SELECT RAISE(ABORT, 'Session name key must not be empty or contain NUL'); END")

def create_v2(path):
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("BEGIN IMMEDIATE")
        for statement in SCHEMA_V2:
            connection.execute(statement)
        connection.execute("INSERT INTO device_settings VALUES (1, 'dark', 30)")
        connection.execute("PRAGMA user_version=2")
        connection.commit()
