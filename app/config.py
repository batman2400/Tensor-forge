"""Runtime settings from the environment. The API key is never given a default."""

from __future__ import annotations

import os
from dataclasses import dataclass

# Decimal megabytes, matching the spec's "about 1 MB / 5 MB / 25 MB".
PREDICT_MAX_BYTES = 1_000_000
BATCH_MAX_BYTES = 5_000_000
JOB_MAX_BYTES = 25_000_000


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _non_negative_env(name: str, default: int) -> int:
    value = _int_env(name, default)
    if value < 0:
        return default
    return value


@dataclass(frozen=True)
class Settings:
    api_key: str | None = None
    port: int = 8000
    job_db_path: str = "/data/jobs.sqlite3"
    inference_threads: int = 2
    log_level: str = "info"
    predict_max_bytes: int = PREDICT_MAX_BYTES
    batch_max_bytes: int = BATCH_MAX_BYTES
    job_max_bytes: int = JOB_MAX_BYTES
    review_threshold: float = 0.5
    job_retention_seconds: int = 12 * 60 * 60
    job_tombstone_seconds: int = 7 * 24 * 60 * 60


def load_settings() -> Settings:
    """Read settings once at process start.

    An empty `API_KEY` is treated as unset. A key that contains whitespace is
    kept as-is; callers must not strip it.
    """
    raw_key = os.environ.get("API_KEY")
    api_key = raw_key if raw_key else None
    return Settings(
        api_key=api_key,
        port=_int_env("PORT", 8000),
        job_db_path=os.environ.get("JOB_DB_PATH") or "/data/jobs.sqlite3",
        inference_threads=max(1, _int_env("INFERENCE_THREADS", 2)),
        log_level=(os.environ.get("LOG_LEVEL") or "info").lower(),
        review_threshold=0.5,
        job_retention_seconds=_non_negative_env("JOB_RETENTION_SECONDS", 12 * 60 * 60),
        job_tombstone_seconds=_non_negative_env("JOB_TOMBSTONE_SECONDS", 7 * 24 * 60 * 60),
    )
