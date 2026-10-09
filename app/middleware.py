"""Pure ASGI contract middleware.

Check order for a protected POST: auth (401) -> method (405) -> content type (415)
-> size (413) -> JSON parse (400). Validation (422) happens in the route.
Unknown paths are 404 JSON with no auth check. A known protected path checks
auth before the method, so a wrong method without a key is 401.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import orjson

from app.auth import authorize
from app.config import Settings
from app.errors import error_body

logger = logging.getLogger("tensorforge.access")

# method set, whether the path requires a key, body limit (POST only)
ROUTES: dict[str, dict[str, Any]] = {
    "/health": {"methods": {"GET", "HEAD"}, "auth": False, "limit": None, "allow": "GET, HEAD"},
    "/predict": {"methods": {"POST"}, "auth": True, "limit": "predict", "allow": "POST"},
    "/predict/batch": {"methods": {"POST"}, "auth": True, "limit": "batch", "allow": "POST"},
    "/batch/jobs": {"methods": {"POST"}, "auth": True, "limit": "job", "allow": "POST"},
    "/": {"methods": {"GET"}, "auth": False, "limit": None, "allow": "GET"},
}

# Job ids are a single path segment. A trailing slash is not a job route.
_JOB_RESULTS_RE = re.compile(r"^/batch/jobs/([^/]+)/results$")
_JOB_ITEM_RE = re.compile(r"^/batch/jobs/([^/]+)$")
_JOB_ITEM = {"methods": {"GET", "DELETE"}, "auth": True, "limit": None, "allow": "GET, DELETE"}
_JOB_RESULTS = {"methods": {"GET"}, "auth": True, "limit": None, "allow": "GET"}


_DEMO = {"methods": {"GET", "HEAD"}, "auth": False, "limit": None, "allow": "GET, HEAD"}


def resolve_route(path: str) -> dict[str, Any] | None:
    found = ROUTES.get(path)
    if found is not None:
        return found
    if path == "/demo" or (path.startswith("/demo/") and ".." not in path):
        return _DEMO
    if _JOB_RESULTS_RE.fullmatch(path):
        return _JOB_RESULTS
    if _JOB_ITEM_RE.fullmatch(path):
        return _JOB_ITEM
    return None


_REQUEST_ID = re.compile(r"^[\x20-\x7e]{1,200}$")
_JSON_TYPE = "application/json"


def header_map(scope: dict[str, Any]) -> dict[str, str]:
    found: dict[str, str] = {}
    for key, value in scope.get("headers") or []:
        name = key.decode("latin-1").lower()
        if name not in found:
            found[name] = value.decode("latin-1")
    return found


def _request_id(headers: dict[str, str]) -> str | None:
    raw = headers.get("x-request-id")
    if raw is None or _REQUEST_ID.fullmatch(raw) is None:
        return None
    return raw


def _is_json(content_type: str | None) -> bool:
    if not content_type:
        return False
    media = content_type.split(";", 1)[0].strip().lower()
    return media == _JSON_TYPE


def _limit_for(settings: Settings, which: str | None) -> int | None:
    if which == "predict":
        return settings.predict_max_bytes
    if which == "batch":
        return settings.batch_max_bytes
    if which == "job":
        return settings.job_max_bytes
    return None


def _replay(body: bytes) -> Any:
    """Return a receive() that yields `body` once. Downstream may read it safely."""
    delivered = False

    async def receive() -> dict[str, Any]:
        nonlocal delivered
        if delivered:
            return {"type": "http.request", "body": b"", "more_body": False}
        delivered = True
        return {"type": "http.request", "body": body, "more_body": False}

    return receive


async def _read_body(receive: Any, limit: int, state: dict[str, bool]) -> bytes | None:
    """Read the body. Return None when the streamed size exceeds `limit`.

    `state["complete"]` becomes true once the client's last body message was seen, so a later
    `_drain` never calls `receive()` again (that would wait for a disconnect).
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            state["complete"] = True
            break
        if message["type"] != "http.request":
            continue
        chunk = message.get("body", b"")
        total += len(chunk)
        if not message.get("more_body", False):
            state["complete"] = True
        if total > limit:
            return None
        chunks.append(chunk)
        if state["complete"]:
            break
    return b"".join(chunks)


# Most we will read and throw away before an early rejection. Slightly above the 25 MB job limit.
DRAIN_CAP_BYTES = 32_000_000


async def _drain(receive: Any, already: int = 0) -> None:
    """Read and discard the rest of the request body, up to `DRAIN_CAP_BYTES`.

    A client that is still uploading when we answer (and hang up) gets a TCP reset and never sees
    our 401/413/415 JSON. Real clients upload over a network, so this matters beyond loopback.
    Past the cap we stop reading and let the server close the connection.
    """
    total = already
    while total <= DRAIN_CAP_BYTES:
        message = await receive()
        if message["type"] == "http.disconnect":
            return
        if message["type"] != "http.request":
            continue
        total += len(message.get("body", b""))
        if not message.get("more_body", False):
            return


def _content_length(headers: dict[str, str]) -> int | None:
    raw = headers.get("content-length")
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


class ContractMiddleware:
    def __init__(self, app: Any, settings: Settings) -> None:
        self.app = app
        self.settings = settings

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started = time.perf_counter()
        status_holder = {"code": 500, "sent": False}
        headers = header_map(scope)
        request_id = _request_id(headers)
        scope["tf_tickets"] = 0
        body_state = {"complete": False}
        path = scope.get("path") or "/"
        method = scope.get("method", "GET").upper()

        async def send_wrapper(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                status_holder["code"] = message["status"]
                status_holder["sent"] = True
                raw_headers = list(message.get("headers") or [])
                if request_id is not None and not any(
                    k.lower() == b"x-request-id" for k, _ in raw_headers
                ):
                    raw_headers.append((b"x-request-id", request_id.encode("ascii")))
                message = {**message, "headers": raw_headers}
            await send(message)

        async def reject(
            status: int, code: str, message: str, extra: dict[str, str] | None = None
        ) -> None:
            # Never answer while the client is mid-upload (see `_drain`).
            if method in {"POST", "PUT", "PATCH", "DELETE"} and not body_state["complete"]:
                await _drain(receive)
                body_state["complete"] = True
            await _send_json(send_wrapper, status, error_body(code, message), extra)

        try:
            route = resolve_route(path)
            if route is None:
                await reject(404, "not_found", "Not found.")
                return

            if route["auth"]:
                ok, why = authorize(headers, self.settings.api_key)
                if not ok:
                    await reject(401, "unauthorized", why, {"WWW-Authenticate": "Bearer"})
                    return

            if method not in route["methods"]:
                await reject(
                    405, "method_not_allowed", "Method not allowed.", {"Allow": route["allow"]}
                )
                return

            if method in {"POST", "PUT", "PATCH"}:
                if not _is_json(headers.get("content-type")):
                    await reject(
                        415, "unsupported_media_type", "Content-Type must be application/json."
                    )
                    return
                limit = _limit_for(self.settings, route["limit"])
                if limit is None:
                    await reject(500, "internal_error", "Internal server error.")
                    return
                declared = _content_length(headers)
                if declared is not None and declared > limit:
                    await reject(
                        413, "payload_too_large", "Request body exceeds the maximum allowed size."
                    )
                    return
                body = await _read_body(receive, limit, body_state)
                if body is None:
                    await reject(
                        413, "payload_too_large", "Request body exceeds the maximum allowed size."
                    )
                    return
                try:
                    payload = orjson.loads(body)
                except orjson.JSONDecodeError:
                    await reject(400, "malformed_json", "Request body is not valid JSON.")
                    return
                scope["tf_payload"] = payload
                receive = _replay(body)

            await self.app(scope, receive, send_wrapper)
        except Exception:
            logger.exception("unhandled middleware error")
            if not status_holder["sent"]:
                await _send_json(
                    send_wrapper,
                    500,
                    error_body("internal_error", "Internal server error."),
                    None,
                )
        finally:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            logger.info(
                orjson.dumps(
                    {
                        "request_id": request_id,
                        "method": method,
                        "path": path,
                        "status": status_holder["code"],
                        "duration_ms": elapsed_ms,
                        "ticket_count": scope.get("tf_tickets", 0),
                    }
                ).decode()
            )


async def _send_json(
    send: Any, status: int, payload: dict[str, Any], extra: dict[str, str] | None
) -> None:
    body = orjson.dumps(payload)
    raw_headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode("ascii")),
    ]
    for key, value in (extra or {}).items():
        raw_headers.append((key.lower().encode("latin-1"), value.encode("latin-1")))
    await send({"type": "http.response.start", "status": status, "headers": raw_headers})
    await send({"type": "http.response.body", "body": body})
