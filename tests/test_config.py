from __future__ import annotations

from app.config import load_settings


def test_empty_api_key_is_unset(monkeypatch):
    monkeypatch.setenv("API_KEY", "")
    assert load_settings().api_key is None


def test_missing_api_key_is_unset(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    assert load_settings().api_key is None


def test_whitespace_in_key_is_kept(monkeypatch):
    monkeypatch.setenv("API_KEY", " ab ")
    assert load_settings().api_key == " ab "


def test_bad_port_falls_back(monkeypatch):
    monkeypatch.setenv("PORT", "nope")
    assert load_settings().port == 8000


def test_inference_threads_at_least_one(monkeypatch):
    monkeypatch.setenv("INFERENCE_THREADS", "0")
    assert load_settings().inference_threads == 1


def test_job_retention_defaults(monkeypatch):
    monkeypatch.delenv("JOB_RETENTION_SECONDS", raising=False)
    monkeypatch.delenv("JOB_TOMBSTONE_SECONDS", raising=False)
    settings = load_settings()
    assert settings.job_retention_seconds == 12 * 60 * 60
    assert settings.job_tombstone_seconds == 7 * 24 * 60 * 60
