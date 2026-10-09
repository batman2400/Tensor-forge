"""Asynchronous batch jobs.

SQLite keeps status and results. Ticket payloads stay in memory only: a crash
or restart marks the job failed, so the text is not needed afterwards. One
worker thread calls `Engine.predict` per ticket and never holds the request
lock while predicting.
"""

from __future__ import annotations

import logging
import sqlite3
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from queue import Empty, Queue
from typing import Any

import orjson

from app.schemas import prediction_to_dict

logger = logging.getLogger("tensorforge.jobs")

MAX_ACTIVE_JOBS = 4  # 1 running + 3 queued
CHUNK_SIZE = 50
YIELD_EVERY = 25
YIELD_SECONDS = 0.005
SHUTDOWN_JOIN_SECONDS = 0.3

INTERRUPTED_MESSAGE = "Service restarted while the job was running."
INTERNAL_MESSAGE = "Internal server error."


@dataclass(frozen=True)
class Accepted:
    body: dict[str, Any]
    replayed: bool


@dataclass(frozen=True)
class JobView:
    """`kind` is `ok`, `missing`, `expired`, or `not_ready`."""

    kind: str
    body: dict[str, Any] | None = None
    status: str | None = None


def _fmt(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def open_database(path: str) -> sqlite3.Connection:
    """Open the job database, falling back to a temp file if `path` is not writable."""
    fallback = Path(tempfile.gettempdir()) / "tensorforge" / "jobs.sqlite3"
    candidates = [Path(path)]
    if candidates[0] != fallback:
        candidates.append(fallback)
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(candidate), check_same_thread=False, timeout=30.0)
        except (OSError, sqlite3.Error) as exc:
            last_error = exc
            logger.info("job database %s is not writable", candidate)
            continue
        conn.row_factory = sqlite3.Row
        conn.isolation_level = None
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=5000")
        except sqlite3.Error:
            logger.exception("job database pragmas failed for %s", candidate)
        if candidate != Path(path):
            logger.info("job database fell back to %s", candidate)
        return conn
    raise RuntimeError("job database is not writable") from last_error


class JobService:
    def __init__(
        self,
        path: str,
        *,
        retention_seconds: int,
        tombstone_seconds: int,
        engine_getter: Callable[[], Any],
    ) -> None:
        self._retention_seconds = retention_seconds
        self._tombstone_seconds = tombstone_seconds
        self._engine_getter = engine_getter
        self._path = path
        self._conn: sqlite3.Connection | None = None
        self._closed = False
        self._lock = threading.Lock()
        self._tickets: dict[str, list[dict[str, Any]]] = {}
        self._cancel: set[str] = set()
        self._queue: Queue[str | None] = Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _ensure_locked(self) -> sqlite3.Connection:
        """Open the database on first use so importing the app does not create files."""
        if self._closed:
            raise RuntimeError("job database is closed")
        if self._conn is None:
            self._conn = open_database(self._path)
            self._migrate()
        return self._conn

    def _migrate(self) -> None:
        assert self._conn is not None
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                total INTEGER NOT NULL,
                processed INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                expires_at TEXT,
                model_version TEXT NOT NULL,
                error_code TEXT,
                error_message TEXT,
                idempotency_key TEXT UNIQUE,
                payload_hash TEXT
            );
            CREATE TABLE IF NOT EXISTS job_results (
                job_id TEXT NOT NULL,
                idx INTEGER NOT NULL,
                prediction_json TEXT NOT NULL,
                PRIMARY KEY (job_id, idx)
            );
            """
        )

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        conn = self._conn
        if conn is None:
            raise RuntimeError("job database is closed")
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield
        except Exception:
            conn.execute("ROLLBACK")
            raise
        else:
            conn.execute("COMMIT")

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                return
            self._ensure_locked()
            self._recover_locked()
            self._stop.clear()
            thread = threading.Thread(target=self._loop, name="job-worker", daemon=True)
            self._thread = thread
        thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._queue.put(None)
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=SHUTDOWN_JOIN_SECONDS)
        with self._lock:
            conn = self._conn
            self._conn = None
            self._closed = True
        if conn is None:
            return
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.close()
        except sqlite3.Error:
            logger.exception("closing job database failed")

    def count_rows(self) -> int:
        with self._lock:
            if self._closed:
                return 0
            row = self._ensure_locked().execute("SELECT COUNT(*) AS n FROM jobs").fetchone()
        return int(row["n"])

    def submit(
        self,
        tickets: list[dict[str, Any]],
        *,
        idempotency_key: str | None,
        payload_hash: str,
        model_version: str,
    ) -> Accepted | None:
        """Insert a queued job. `None` means the active-job capacity is full."""
        stored = [dict(ticket) for ticket in tickets]
        with self._lock:
            self._ensure_locked()
            self._sweep_locked()
            if idempotency_key:
                existing = self._find_key_locked(idempotency_key)
                if existing is not None:
                    return Accepted(_status_from_row(existing), replayed=True)
            active = self._conn.execute(
                "SELECT COUNT(*) AS n FROM jobs WHERE status IN ('queued', 'running')"
            ).fetchone()
            if int(active["n"]) >= MAX_ACTIVE_JOBS:
                return None
            job_id = str(uuid.uuid4())
            created_at = _fmt(datetime.now(UTC))
            try:
                with self._transaction():
                    self._conn.execute(
                        """
                        INSERT INTO jobs (
                            job_id, status, total, processed, created_at, started_at,
                            finished_at, expires_at, model_version, error_code,
                            error_message, idempotency_key, payload_hash
                        ) VALUES (?, 'queued', ?, 0, ?, NULL, NULL, NULL, ?, NULL, NULL, ?, ?)
                        """,
                        (
                            job_id,
                            len(stored),
                            created_at,
                            model_version,
                            idempotency_key,
                            payload_hash,
                        ),
                    )
            except sqlite3.IntegrityError:
                if idempotency_key:
                    existing = self._find_key_locked(idempotency_key)
                    if existing is not None:
                        return Accepted(_status_from_row(existing), replayed=True)
                raise
            self._tickets[job_id] = stored
            body = {
                "job_id": job_id,
                "status": "queued",
                "total": len(stored),
                "processed": 0,
                "created_at": created_at,
                "started_at": None,
                "finished_at": None,
                "expires_at": None,
                "model_version": model_version,
                "error": None,
            }
        self._queue.put(job_id)
        logger.info("job accepted id=%s total=%s", job_id, len(stored))
        return Accepted(body, replayed=False)

    def get(self, job_id: str) -> JobView:
        with self._lock:
            if self._closed:
                return JobView("missing")
            self._ensure_locked()
            self._sweep_locked()
            row = self._conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                return JobView("missing")
            if row["status"] == "expired":
                return JobView("expired")
            return JobView("ok", _status_from_row(row))

    def delete(self, job_id: str) -> bool:
        with self._lock:
            if self._closed:
                return False
            self._ensure_locked()
            row = self._conn.execute(
                "SELECT status FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if row is None:
                return False
            if row["status"] in {"queued", "running"}:
                self._cancel.add(job_id)
            self._tickets.pop(job_id, None)
            with self._transaction():
                self._conn.execute("DELETE FROM job_results WHERE job_id = ?", (job_id,))
                self._conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
            return True

    def results(self, job_id: str, offset: int, limit: int | None) -> JobView:
        with self._lock:
            if self._closed:
                return JobView("missing")
            self._ensure_locked()
            self._sweep_locked()
            row = self._conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                return JobView("missing")
            if row["status"] == "expired":
                return JobView("expired")
            if row["status"] != "succeeded":
                return JobView("not_ready", status=str(row["status"]))
            total = int(row["total"])
            remaining = max(0, total - offset)
            page = min(remaining, 5000) if limit is None else limit
            take = min(page, remaining)
            fetched = self._conn.execute(
                """
                SELECT prediction_json FROM job_results
                WHERE job_id = ? ORDER BY idx LIMIT ? OFFSET ?
                """,
                (job_id, take, offset),
            ).fetchall()
            model_version = str(row["model_version"])
        predictions = [orjson.loads(item["prediction_json"]) for item in fetched]
        next_offset = offset + take if offset + take < total else None
        body = {
            "job_id": job_id,
            "status": "succeeded",
            "total": total,
            "offset": offset,
            "limit": take if limit is None else limit,
            "next_offset": next_offset,
            "model_version": model_version,
            "predictions": predictions,
        }
        return JobView("ok", body)

    def _find_key_locked(self, idempotency_key: str) -> sqlite3.Row | None:
        assert self._conn is not None
        return self._conn.execute(
            """
            SELECT * FROM jobs
            WHERE idempotency_key = ? AND status != 'expired'
            """,
            (idempotency_key,),
        ).fetchone()

    def _recover_locked(self) -> None:
        if self._conn is None:
            return
        finished_at, expires_at = self._finish_stamps()
        with self._transaction():
            self._conn.execute(
                """
                UPDATE jobs
                SET status = 'failed',
                    error_code = 'interrupted',
                    error_message = ?,
                    finished_at = ?,
                    expires_at = ?
                WHERE status IN ('queued', 'running')
                """,
                (INTERRUPTED_MESSAGE, finished_at, expires_at),
            )

    def _finish_stamps(self) -> tuple[str, str]:
        now = datetime.now(UTC)
        return _fmt(now), _fmt(now + timedelta(seconds=self._retention_seconds))

    def _sweep_locked(self) -> None:
        conn = self._conn
        if conn is None:
            return
        now = datetime.now(UTC)
        stamp = _fmt(now)
        rows = conn.execute(
            """
            SELECT job_id FROM jobs
            WHERE status IN ('succeeded', 'failed')
              AND expires_at IS NOT NULL
              AND expires_at <= ?
            """,
            (stamp,),
        ).fetchall()
        if rows:
            with self._transaction():
                for row in rows:
                    job_id = str(row["job_id"])
                    conn.execute("DELETE FROM job_results WHERE job_id = ?", (job_id,))
                    conn.execute(
                        """
                        UPDATE jobs
                        SET status = 'expired', idempotency_key = NULL
                        WHERE job_id = ?
                        """,
                        (job_id,),
                    )
                    self._tickets.pop(job_id, None)
        purge_before = _fmt(now - timedelta(seconds=self._tombstone_seconds))
        conn.execute(
            """
            DELETE FROM jobs
            WHERE status = 'expired' AND expires_at IS NOT NULL AND expires_at <= ?
            """,
            (purge_before,),
        )

    def _sweep(self) -> None:
        with self._lock:
            self._sweep_locked()

    def _loop(self) -> None:
        while True:
            try:
                job_id = self._queue.get(timeout=0.2)
            except Empty:
                if self._stop.is_set():
                    return
                try:
                    self._sweep()
                except Exception:
                    logger.exception("job sweep failed")
                continue
            if job_id is None:
                if self._stop.is_set():
                    return
                continue
            try:
                self._run_one(job_id)
            except Exception:
                logger.exception("job worker error id=%s", job_id)
                self._fail(job_id, "internal_error", INTERNAL_MESSAGE)

    def _run_one(self, job_id: str) -> None:
        if self._stop.is_set():
            return
        with self._lock:
            if self._conn is None:
                return
            tickets = self._tickets.get(job_id)
            row = self._conn.execute(
                "SELECT status FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if row is None or row["status"] != "queued":
                self._tickets.pop(job_id, None)
                self._cancel.discard(job_id)
                return
            if tickets is None:
                self._tickets.pop(job_id, None)
                missing = True
            else:
                missing = False
                with self._transaction():
                    self._conn.execute(
                        """
                        UPDATE jobs SET status = 'running', started_at = ?
                        WHERE job_id = ? AND status = 'queued'
                        """,
                        (_fmt(datetime.now(UTC)), job_id),
                    )
        if missing or tickets is None:
            self._fail(job_id, "internal_error", INTERNAL_MESSAGE)
            return
        engine = self._engine_getter()
        if engine is None:
            self._fail(job_id, "internal_error", INTERNAL_MESSAGE)
            return
        chunk: list[tuple[int, dict[str, Any]]] = []
        try:
            for index, ticket in enumerate(tickets):
                if not self._still_active(job_id):
                    return
                prediction = dict(engine.predict(ticket))
                ticket_id = ticket.get("ticket_id")
                if ticket_id is not None:
                    prediction["ticket_id"] = ticket_id
                chunk.append((index, prediction_to_dict(prediction)))
                done = index + 1
                if len(chunk) >= CHUNK_SIZE or done == len(tickets):
                    if not self._flush(job_id, chunk, done):
                        return
                    chunk = []
                if done % YIELD_EVERY == 0:
                    if not self._still_active(job_id):
                        return
                    time.sleep(YIELD_SECONDS)
            self._succeed(job_id)
        except Exception:
            logger.exception("job failed id=%s", job_id)
            self._fail(job_id, "internal_error", INTERNAL_MESSAGE)
        finally:
            with self._lock:
                self._tickets.pop(job_id, None)
                self._cancel.discard(job_id)

    def _still_active(self, job_id: str) -> bool:
        if self._stop.is_set():
            return False
        with self._lock:
            if self._conn is None or job_id in self._cancel:
                return False
            row = self._conn.execute(
                "SELECT status FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        return row is not None and row["status"] == "running"

    def _flush(self, job_id: str, chunk: list[tuple[int, dict[str, Any]]], processed: int) -> bool:
        payload = [(job_id, idx, orjson.dumps(body).decode()) for idx, body in chunk]
        with self._lock:
            if self._conn is None or job_id in self._cancel:
                return False
            row = self._conn.execute(
                "SELECT status FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if row is None or row["status"] != "running":
                return False
            with self._transaction():
                self._conn.executemany(
                    "INSERT INTO job_results (job_id, idx, prediction_json) VALUES (?, ?, ?)",
                    payload,
                )
                self._conn.execute(
                    """
                    UPDATE jobs SET processed = MAX(processed, ?)
                    WHERE job_id = ? AND status = 'running'
                    """,
                    (processed, job_id),
                )
            return True

    def _succeed(self, job_id: str) -> None:
        finished_at, expires_at = self._finish_stamps()
        with self._lock:
            if self._conn is None:
                return
            with self._transaction():
                self._conn.execute(
                    """
                    UPDATE jobs
                    SET status = 'succeeded',
                        processed = total,
                        finished_at = ?,
                        expires_at = ?,
                        error_code = NULL,
                        error_message = NULL
                    WHERE job_id = ? AND status = 'running'
                    """,
                    (finished_at, expires_at, job_id),
                )

    def _fail(self, job_id: str, code: str, message: str) -> None:
        # `message` is a fixed sentence. Do not pass exception text through.
        finished_at, expires_at = self._finish_stamps()
        try:
            with self._lock:
                if self._conn is None:
                    return
                with self._transaction():
                    self._conn.execute(
                        """
                        UPDATE jobs
                        SET status = 'failed',
                            error_code = ?,
                            error_message = ?,
                            finished_at = ?,
                            expires_at = ?
                        WHERE job_id = ? AND status IN ('queued', 'running')
                        """,
                        (code, message, finished_at, expires_at, job_id),
                    )
                self._tickets.pop(job_id, None)
        except sqlite3.Error:
            logger.exception("could not mark job failed id=%s", job_id)


def _status_from_row(row: sqlite3.Row) -> dict[str, Any]:
    error = None
    if row["error_code"]:
        error = {"code": row["error_code"], "message": row["error_message"]}
    return {
        "job_id": row["job_id"],
        "status": row["status"],
        "total": row["total"],
        "processed": row["processed"],
        "created_at": row["created_at"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "expires_at": row["expires_at"],
        "model_version": row["model_version"],
        "error": error,
    }
