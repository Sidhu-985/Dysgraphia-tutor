"""
Storage
========
Two responsibilities, deliberately kept in one small module so every other
file only ever imports `SessionStore` / `AttemptLogStore` and never touches
SQLite/dict details directly:

  1. `SessionStore`      - in-memory, process-local `SessionContext` per
                            active session (fast path for the hot request
                            loop: preprocess -> features -> state -> RL ->
                            safety -> intervention).
  2. `AttemptLogStore`    - durable (SQLite) record of every attempt: raw
                            feature dict, state vector, chosen/overridden
                            action, and individual reward components, so
                            training-signal analysis and the reward-hacking
                            audit (dysgraphia_rl.utils.reward_utils
                            .audit_action_distribution) can be run on real
                            interaction data later, per project spec
                            Section 8 ("log individual reward components").

This is a research-prototype-scale store (SQLite, single file). Swapping
it for a real database later means editing only this file.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from .config import settings
from .state.session_context import SessionContext


class SessionStore:
    """Process-local session state. NOT durable across restarts by design
    -- a session in progress is expected to be short-lived; the durable
    record is `AttemptLogStore`, written on every submitted attempt."""

    def __init__(self):
        self._sessions: dict[str, SessionContext] = {}
        self._lock = threading.Lock()

    def create(self, child_id: str) -> SessionContext:
        session_id = str(uuid.uuid4())
        ctx = SessionContext(session_id=session_id, child_id=child_id)
        with self._lock:
            self._sessions[session_id] = ctx
        return ctx

    def get(self, session_id: str) -> Optional[SessionContext]:
        with self._lock:
            return self._sessions.get(session_id)

    def save(self, session: SessionContext) -> None:
        with self._lock:
            self._sessions[session.session_id] = session

    def end(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].ended = True


class AttemptLogStore:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path or settings.storage.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    child_id TEXT NOT NULL,
                    exercise_id TEXT,
                    attempt_index INTEGER,
                    created_at REAL,
                    static_features TEXT,
                    dynamic_features TEXT,
                    state_vector TEXT,
                    quality_score REAL,
                    raw_action_id INTEGER,
                    safe_action_id INTEGER,
                    safety_overridden INTEGER,
                    safety_reasons TEXT,
                    reward REAL,
                    reward_components TEXT
                )
                """
            )
            # Columns added after the original schema. ALTER is used so an
            # existing database file keeps its rows (never dropped).
            existing = {r[1] for r in conn.execute("PRAGMA table_info(attempts)").fetchall()}
            for col, decl in (("data_source", "TEXT"), ("policy_source", "TEXT"),
                              ("pressure_available", "INTEGER"), ("n_points", "INTEGER"),
                              ("n_strokes", "INTEGER"), ("raw_points", "TEXT")):
                if col not in existing:
                    try:
                        conn.execute(f"ALTER TABLE attempts ADD COLUMN {col} {decl}")
                    except sqlite3.OperationalError as exc:  # another process migrated first
                        if "duplicate column" not in str(exc):
                            raise
            # One row per RL transition (s_t, a_t, r_t, s_{t+1}, done).
            # s_{t+1} and r_t come from the NEXT observed attempt. When a
            # session ends before another attempt, the final decision is
            # stored with reward NULL and outcome_observed=0. No outcome is
            # ever invented.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS transitions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    child_id TEXT NOT NULL,
                    data_source TEXT,
                    decision_attempt_index INTEGER,
                    outcome_attempt_index INTEGER,
                    state TEXT NOT NULL,
                    action INTEGER NOT NULL,
                    raw_action INTEGER,
                    reward REAL,
                    reward_components TEXT,
                    next_state TEXT,
                    done INTEGER NOT NULL,
                    outcome_observed INTEGER NOT NULL,
                    created_at REAL
                )
                """
            )
            conn.commit()

    def log_attempt(self, record: dict) -> int:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO attempts (
                    session_id, child_id, exercise_id, attempt_index, created_at,
                    static_features, dynamic_features, state_vector, quality_score,
                    raw_action_id, safe_action_id, safety_overridden, safety_reasons,
                    reward, reward_components, data_source, policy_source,
                    pressure_available, n_points, n_strokes, raw_points
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["session_id"], record["child_id"], record.get("exercise_id"),
                    record.get("attempt_index"), record.get("created_at", time.time()),
                    json.dumps(record.get("static_features", {})),
                    json.dumps(record.get("dynamic_features", {})),
                    json.dumps(record.get("state_vector", [])),
                    record.get("quality_score"),
                    record.get("raw_action_id"), record.get("safe_action_id"),
                    int(record.get("safety_overridden", False)),
                    json.dumps(record.get("safety_reasons", [])),
                    record.get("reward"),
                    json.dumps(record.get("reward_components", {})),
                    record.get("data_source"), record.get("policy_source"),
                    None if record.get("pressure_available") is None else int(record["pressure_available"]),
                    record.get("n_points"), record.get("n_strokes"),
                    json.dumps(record["raw_points"]) if record.get("raw_points") is not None else None,
                ),
            )
            conn.commit()
            return cur.lastrowid

    def log_transition(self, record: dict) -> int:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO transitions (
                    session_id, child_id, data_source, decision_attempt_index,
                    outcome_attempt_index, state, action, raw_action, reward,
                    reward_components, next_state, done, outcome_observed, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["session_id"], record["child_id"], record.get("data_source"),
                    record.get("decision_attempt_index"), record.get("outcome_attempt_index"),
                    json.dumps(record["state"]), int(record["action"]),
                    record.get("raw_action"), record.get("reward"),
                    json.dumps(record.get("reward_components")) if record.get("reward_components") is not None else None,
                    json.dumps(record["next_state"]) if record.get("next_state") is not None else None,
                    int(record["done"]), int(record["outcome_observed"]),
                    record.get("created_at", time.time()),
                ),
            )
            conn.commit()
            return cur.lastrowid

    def get_session_transitions(self, session_id: str) -> list[dict]:
        with self._lock, self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM transitions WHERE session_id = ? ORDER BY id ASC", (session_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_session_attempts(self, session_id: str) -> list[dict]:
        with self._lock, self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM attempts WHERE session_id = ? ORDER BY id ASC", (session_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_action_log(self, session_id: Optional[str] = None) -> list[int]:
        with self._lock, self._connect() as conn:
            if session_id:
                rows = conn.execute(
                    "SELECT safe_action_id FROM attempts WHERE session_id = ?", (session_id,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT safe_action_id FROM attempts").fetchall()
            return [r[0] for r in rows if r[0] is not None]


_session_store: Optional[SessionStore] = None
_attempt_log_store: Optional[AttemptLogStore] = None
# Phase E fix: FastAPI runs sync endpoints in a thread pool. Without this
# lock, the first concurrent requests could each build their OWN
# SessionStore, so sessions vanished between requests, and their own
# AttemptLogStore, which raced on the schema migration ("duplicate column
# name"). Found by the 8-thread test in scripts/phase_e_failure_modes.py.
_singleton_lock = threading.Lock()


def get_session_store() -> SessionStore:
    global _session_store
    if _session_store is None:
        with _singleton_lock:
            if _session_store is None:
                _session_store = SessionStore()
    return _session_store


def get_attempt_log_store() -> AttemptLogStore:
    global _attempt_log_store
    if _attempt_log_store is None:
        with _singleton_lock:
            if _attempt_log_store is None:
                _attempt_log_store = AttemptLogStore()
    return _attempt_log_store
