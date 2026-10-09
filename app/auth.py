"""API key extraction and constant-time comparison."""

from __future__ import annotations

import hmac
import re
from collections.abc import Mapping

_BEARER = re.compile(r"(?i)bearer (.*)$")

MISSING_MESSAGE = "Missing API key. Send X-API-Key or Authorization Bearer."
INVALID_MESSAGE = "Invalid API key."


def _equals(presented: str, expected: str) -> bool:
    return hmac.compare_digest(presented.encode("utf-8"), expected.encode("utf-8"))


def presented_keys(headers: Mapping[str, str]) -> list[str]:
    """Keys the client actually sent. Empty when neither header is present."""
    found: list[str] = []
    if "x-api-key" in headers:
        found.append(headers["x-api-key"])
    authorization = headers.get("authorization")
    if authorization is not None:
        match = _BEARER.match(authorization)
        # A non-Bearer Authorization header is still an attempt, so it must not
        # take the "missing key" path.
        found.append(match.group(1) if match else "")
    return found


def authorize(headers: Mapping[str, str], expected: str | None) -> tuple[bool, str]:
    """Return (accepted, message). Message is only meaningful when accepted is False.

    Every presented key is compared, so a matching first header does not return early.
    An unset expected key rejects everything (the service must not run open).
    """
    presented = presented_keys(headers)
    if not expected:
        return False, INVALID_MESSAGE if presented else MISSING_MESSAGE
    if not presented:
        return False, MISSING_MESSAGE
    ok = False
    for candidate in presented:
        ok = _equals(candidate, expected) or ok
    return ok, "" if ok else INVALID_MESSAGE
