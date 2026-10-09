"""An early rejection must not be sent while the client is still uploading.

Over a real network a client that is mid-upload when the server answers and hangs up gets a TCP
reset and never sees the JSON error. The middleware therefore reads (and discards) the rest of
the body first. These tests drive the ASGI app directly with a scripted, chunked receive().
"""

from __future__ import annotations

import asyncio
import json

import pytest

from app.config import Settings
from app.middleware import ContractMiddleware
from tests.helpers import API_KEY

CHUNK = 16_384


async def _ok_app(scope, receive, send):  # pragma: no cover - never reached for rejections
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"{}"})


def _run(headers: list[tuple[bytes, bytes]], total: int, *, path="/predict", chunk_size=CHUNK):
    """Send `total` body bytes in chunks; return (status, chunks_left_when_response_started)."""
    settings = Settings(api_key=API_KEY, predict_max_bytes=1000, job_max_bytes=10_000)
    app = ContractMiddleware(_ok_app, settings)
    chunks = [b"x" * min(chunk_size, total - i) for i in range(0, total, chunk_size)] or [b""]
    state = {"left": len(chunks), "status": None, "left_at_response": None, "extra_receive": 0}

    async def receive():
        if state["left"] == 0:
            state["extra_receive"] += 1
            return {"type": "http.disconnect"}
        state["left"] -= 1
        return {
            "type": "http.request",
            "body": chunks[len(chunks) - state["left"] - 1],
            "more_body": state["left"] > 0,
        }

    async def send(message):
        if message["type"] == "http.response.start":
            state["status"] = message["status"]
            state["left_at_response"] = state["left"]

    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": headers,
        "query_string": b"",
    }
    asyncio.run(app(scope, receive, send))
    return state


def _headers(*, key: str | None = API_KEY, content_type="application/json", length=None):
    out = []
    if content_type:
        out.append((b"content-type", content_type.encode()))
    if key is not None:
        out.append((b"x-api-key", key.encode()))
    if length is not None:
        out.append((b"content-length", str(length).encode()))
    return out


@pytest.mark.parametrize(
    ("label", "headers_kwargs", "expected"),
    [
        ("too large (declared)", {"length": 200_000}, 413),
        ("too large (undeclared / chunked)", {}, 413),
        ("wrong key", {"key": "nope", "length": 200_000}, 401),
        ("no key", {"key": None, "length": 200_000}, 401),
        ("bad content type", {"content_type": "text/plain", "length": 200_000}, 415),
    ],
)
def test_body_is_fully_read_before_rejecting(label, headers_kwargs, expected):
    state = _run(_headers(**headers_kwargs), 200_000)
    assert state["status"] == expected, label
    assert state["left_at_response"] == 0, f"{label}: answered with the upload unfinished"
    assert state["extra_receive"] == 0, f"{label}: read past the end of the body"


def test_malformed_json_does_not_wait_for_more_input():
    # body fully read by the parser; draining again would block until disconnect
    settings = Settings(api_key=API_KEY, predict_max_bytes=1000)
    app = ContractMiddleware(_ok_app, settings)
    calls = {"n": 0}
    status = {}

    async def receive():
        calls["n"] += 1
        if calls["n"] > 1:
            raise AssertionError("receive() called again after the body was complete")
        return {"type": "http.request", "body": b"{{{", "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            status["code"] = message["status"]
        elif message["type"] == "http.response.body":
            status["body"] = json.loads(message["body"])

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/predict",
        "headers": _headers(length=3),
        "query_string": b"",
    }
    asyncio.run(app(scope, receive, send))
    assert status["code"] == 400
    assert status["body"]["error"]["code"] == "malformed_json"


def test_overflow_in_final_chunk_does_not_hang():
    # the chunk that crosses the limit is also the last one
    state = _run(_headers(), 1001, chunk_size=2000)
    assert state["status"] == 413
    assert state["extra_receive"] == 0


def test_drain_is_bounded(monkeypatch):
    import app.middleware as mw

    monkeypatch.setattr(mw, "DRAIN_CAP_BYTES", 50_000)
    state = _run(_headers(length=1_000_000), 1_000_000)
    assert state["status"] == 413
    assert state["left_at_response"] > 0, "must stop reading at the cap, not read everything"


def test_access_log_reports_body_and_service_time(caplog):
    caplog.set_level("INFO", logger="tensorforge.access")
    _run(_headers(key=None, length=200_000), 200_000)
    entries = [json.loads(r.getMessage()) for r in caplog.records if r.name == "tensorforge.access"]
    assert entries, "no access log entry"
    entry = entries[-1]
    assert entry["status"] == 401
    assert entry["body_ms"] >= 0
    assert entry["service_ms"] == entry["duration_ms"] - entry["body_ms"]
