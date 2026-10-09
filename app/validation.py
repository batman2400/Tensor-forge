"""Hand-written validators. They mirror the JSON Schemas and collect every error.

Duplicate `ticket_id` values are rejected here. JSON Schema cannot express that
constraint; the differential test treats it as an intentional extra rule.
"""

from __future__ import annotations

import re
from typing import Any

from app.labels import CHANNELS

_NON_SPACE = re.compile(r"\S")
_CHANNELS = frozenset(CHANNELS)

TEXT_NONEMPTY = "must contain at least one non-whitespace character"
CHANNEL_ENUM = "must be one of email, chat, call_transcript"


def _ticket_errors(obj: dict[str, Any], *, require_ticket_id: bool) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []

    if require_ticket_id and "ticket_id" not in obj:
        errors.append({"field": "ticket_id", "issue": "is required"})
    elif "ticket_id" in obj:
        ticket_id = obj["ticket_id"]
        if not isinstance(ticket_id, str):
            errors.append({"field": "ticket_id", "issue": "must be a string"})
        elif len(ticket_id) > 64:
            errors.append({"field": "ticket_id", "issue": "must be at most 64 characters"})

    if "channel" not in obj:
        errors.append({"field": "channel", "issue": "is required"})
    else:
        channel = obj["channel"]
        if not isinstance(channel, str):
            errors.append({"field": "channel", "issue": "must be a string"})
        elif channel not in _CHANNELS:
            errors.append({"field": "channel", "issue": CHANNEL_ENUM})

    if "subject" in obj:
        subject = obj["subject"]
        if not isinstance(subject, str):
            errors.append({"field": "subject", "issue": "must be a string"})
        elif len(subject) > 500:
            errors.append({"field": "subject", "issue": "must be at most 500 characters"})

    if "text" not in obj:
        errors.append({"field": "text", "issue": "is required"})
    else:
        text = obj["text"]
        if not isinstance(text, str):
            errors.append({"field": "text", "issue": "must be a string"})
        else:
            if len(text) > 10000:
                errors.append({"field": "text", "issue": "must be at most 10000 characters"})
            if _NON_SPACE.search(text) is None:
                errors.append({"field": "text", "issue": TEXT_NONEMPTY})

    return errors


def validate_single(obj: Any) -> list[dict[str, Any]]:
    """Errors for a `POST /predict` body. No `index` keys."""
    if not isinstance(obj, dict):
        return [{"field": "body", "issue": "must be a JSON object"}]
    return _ticket_errors(obj, require_ticket_id=False)


def validate_batch(obj: Any, max_items: int) -> list[dict[str, Any]]:
    """Errors for a batch envelope. Item errors carry a zero-based `index`."""
    if not isinstance(obj, dict):
        return [{"field": "body", "issue": "must be a JSON object"}]
    if "tickets" not in obj:
        return [{"field": "tickets", "issue": "is required"}]
    tickets = obj["tickets"]
    if not isinstance(tickets, list):
        return [{"field": "tickets", "issue": "must be an array"}]

    errors: list[dict[str, Any]] = []
    if not 1 <= len(tickets) <= max_items:
        errors.append(
            {
                "field": "tickets",
                "issue": f"must contain between 1 and {max_items} items",
            }
        )

    seen: dict[str, int] = {}
    for index, item in enumerate(tickets):
        if not isinstance(item, dict):
            errors.append({"index": index, "field": "tickets", "issue": "must be an object"})
            continue
        for problem in _ticket_errors(item, require_ticket_id=True):
            errors.append({"index": index, **problem})
        ticket_id = item.get("ticket_id")
        if isinstance(ticket_id, str):
            if ticket_id in seen:
                errors.append(
                    {
                        "index": index,
                        "field": "ticket_id",
                        "issue": "duplicate of an earlier item",
                    }
                )
            else:
                seen[ticket_id] = index
    return errors
