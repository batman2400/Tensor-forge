"""Our validators agree with the JSON Schemas, plus the duplicate-id rule."""

from __future__ import annotations

import random

from app.labels import CHANNELS
from app.validation import CHANNEL_ENUM, TEXT_NONEMPTY, validate_batch, validate_single
from tests.helpers import validator


def _single_ok(payload) -> bool:
    return validator("predict_request").is_valid(payload)


def _batch_ok(payload) -> bool:
    return validator("batch_request").is_valid(payload)


def test_messages_for_the_spec_examples():
    empty = validate_single({"channel": "chat", "text": ""})
    assert empty[0]["field"] == "text"
    assert empty[0]["issue"] == TEXT_NONEMPTY

    channel = validate_single({"channel": "sms", "text": "hi"})
    assert channel[0]["issue"] == CHANNEL_ENUM

    whitespace = validate_single({"channel": "chat", "text": " \n\t "})
    assert any(item["issue"] == TEXT_NONEMPTY for item in whitespace)

    assert validate_single({"channel": "chat", "text": "🙏"}) == []
    assert validate_single({"channel": "email", "text": "hi", "language": "en"}) == []
    assert validate_single({"channel": "chat", "text": "a" * 10000}) == []
    assert validate_single({"channel": "chat", "text": "a" * 10001})
    assert validate_single({"channel": "email", "subject": "s" * 500, "text": "hi"}) == []
    assert validate_single({"channel": "email", "subject": "s" * 501, "text": "hi"})
    assert validate_single({"channel": "email", "subject": None, "text": "hi"})


def test_batch_reports_every_index():
    payload = {
        "tickets": [
            {"ticket_id": "a", "channel": "chat", "text": "ok"},
            {"ticket_id": "b", "channel": "chat", "text": "  "},
            {"channel": "chat", "text": "ok"},
            "nope",
        ]
    }
    errors = validate_batch(payload, max_items=100)
    indexes = [item["index"] for item in errors]
    assert 1 in indexes
    assert 2 in indexes
    assert 3 in indexes
    assert 0 not in indexes


def test_duplicates_are_stricter_than_the_schema():
    payload = {
        "tickets": [
            {"ticket_id": "same", "channel": "chat", "text": "one"},
            {"ticket_id": "same", "channel": "chat", "text": "two"},
        ]
    }
    assert _batch_ok(payload)
    errors = validate_batch(payload, max_items=100)
    assert errors[-1]["index"] == 1
    assert errors[-1]["field"] == "ticket_id"


def _mutate(rng: random.Random, value):
    roll = rng.randrange(8)
    if roll == 0:
        return None
    if roll == 1:
        return rng.randrange(3)
    if roll == 2:
        return []
    if roll == 3:
        return {}
    if roll == 4:
        return "x" * rng.choice([0, 1, 64, 65, 500, 501, 10000, 10001])
    if roll == 5:
        return "   "
    if roll == 6:
        return True
    return value


def test_single_matches_jsonschema():
    rng = random.Random(0)
    cases = [
        {"channel": "chat", "text": "hi"},
        {"channel": "email", "subject": "Hello", "text": "body", "ticket_id": "id-1"},
        {"channel": "chat", "text": ""},
        {"channel": "chat", "text": " \n "},
        {"channel": "sms", "text": "hi"},
        {"text": "hi"},
        {"channel": "chat"},
        {"channel": "chat", "text": 5},
        {"channel": "chat", "text": "a" * 10001},
        {"channel": "chat", "text": "a" * 10000},
        {"channel": None, "text": "hi"},
        {"channel": "email", "subject": None, "text": "hi"},
        {"channel": "email", "subject": "s" * 501, "text": "hi"},
        {"channel": "chat", "text": "hi", "ticket_id": "t" * 65},
        {"channel": "chat", "text": "hi", "ticket_id": "t" * 64},
        [],
        None,
        "hi",
        1,
        {"channel": "chat", "text": "hi", "language": "si", "meta": {"a": [1, {"b": True}]}},
        {"channel": "call_transcript", "text": "🙏"},
    ]
    template = {"channel": "chat", "text": "hello there", "subject": "sub", "ticket_id": "abc"}
    for _ in range(300):
        mutated = dict(template)
        key = rng.choice(["channel", "text", "subject", "ticket_id"])
        if rng.random() < 0.2:
            mutated.pop(key, None)
        else:
            mutated[key] = _mutate(rng, mutated[key])
        if rng.random() < 0.1:
            mutated["language"] = rng.choice(["en", 1, None])
        cases.append(mutated)
        cases.append(_mutate(rng, template))

    mismatches = []
    for case in cases:
        schema_ok = _single_ok(case)
        ours_ok = validate_single(case) == []
        if schema_ok != ours_ok:
            mismatches.append((schema_ok, ours_ok, case))
    assert not mismatches, mismatches[:5]


def test_batch_matches_jsonschema_except_duplicates():
    rng = random.Random(1)

    def item(ticket_id, **overrides):
        body = {
            "ticket_id": ticket_id,
            "channel": rng.choice(list(CHANNELS)),
            "text": f"text {ticket_id}",
        }
        body.update(overrides)
        return body

    cases = [
        {"tickets": [item("a"), item("b")]},
        {"tickets": []},
        {"tickets": [item(f"id{i}") for i in range(100)]},
        {"tickets": [item(f"id{i}") for i in range(101)]},
        {"items": [item("a")]},
        [],
        {"tickets": "no"},
        {"tickets": [item("a"), {"channel": "chat", "text": "hi"}]},
        {"tickets": [item("a"), "bad"]},
        {"tickets": [item("a", text="")]},
        {"tickets": [item("a", channel="sms")]},
        {"tickets": [item("a")], "note": "ignored"},
    ]
    for _ in range(120):
        count = rng.randint(0, 4)
        tickets = [item(f"r{i}-{rng.randrange(10000)}") for i in range(count)]
        if tickets and rng.random() < 0.5:
            target = tickets[rng.randrange(len(tickets))]
            field = rng.choice(["text", "channel", "subject", "ticket_id"])
            if rng.random() < 0.3:
                target.pop(field, None)
            else:
                target[field] = _mutate(rng, target.get(field, "x"))
        payload = {"tickets": tickets}
        if rng.random() < 0.05:
            payload = _mutate(rng, payload)
        cases.append(payload)

    mismatches = []
    compared = 0
    for case in cases:
        if isinstance(case, dict) and isinstance(case.get("tickets"), list):
            ids = [
                entry.get("ticket_id")
                for entry in case["tickets"]
                if isinstance(entry, dict) and isinstance(entry.get("ticket_id"), str)
            ]
            if len(ids) != len(set(ids)):
                continue
        schema_ok = _batch_ok(case)
        ours_ok = validate_batch(case, max_items=100) == []
        compared += 1
        if schema_ok != ours_ok:
            mismatches.append((schema_ok, ours_ok, case, validate_batch(case, max_items=100)))
    assert compared >= 100
    assert not mismatches, mismatches[:3]
