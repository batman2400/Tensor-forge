"""Contract checks for /health, /predict and /predict/batch."""

from __future__ import annotations

import asyncio

import orjson

from app.config import PREDICT_MAX_BYTES
from app.middleware import ContractMiddleware
from tests.helpers import API_KEY, BoomEngine, assert_error, assert_schema, auth, client_for


def _predict(client, payload, headers=None, content=None):
    if content is not None:
        return client.post("/predict", content=content, headers=headers)
    return client.post("/predict", json=payload, headers=headers)


def test_health_and_predict_share_model_version(client):
    health = client.get("/health")
    assert health.status_code == 200
    assert_schema("health_response", health.json())
    assert health.json()["status"] == "ok"
    assert health.json()["model_loaded"] is True

    response = client.post(
        "/predict",
        json={"channel": "chat", "text": "Where is my driver?"},
        headers=auth(),
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert_schema("predict_response", body)
    assert body["model_version"] == health.json()["model_version"]
    assert body["team"] == "Front-line Support"
    assert body["secondary_category"] is None
    assert "ticket_id" not in body


def test_head_health(client):
    get_response = client.get("/health")
    head_response = client.head("/health")
    assert head_response.status_code == get_response.status_code


def test_root_is_json(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert body["service"] == "tensorforge"
    assert body["model_loaded"] is True


def test_docs_and_unknown_routes_are_json_404(client):
    for path in ("/docs", "/redoc", "/openapi.json", "/nope", "/predict/"):
        response = client.post(path, json={"channel": "chat", "text": "hi"})
        body = assert_error(response, 404, "not_found")
        assert body["error"]["code"] == "not_found"


def test_unknown_route_does_not_require_a_key(client):
    response = client.get("/missing")
    assert_error(response, 404, "not_found")


def test_wrong_method_on_health_is_405(client):
    response = client.post("/health", json={"channel": "chat", "text": "hi"})
    assert_error(response, 405, "method_not_allowed")
    assert response.headers["allow"] == "GET, HEAD"


def test_wrong_method_on_predict_checks_auth_first(client):
    missing = client.get("/predict")
    assert_error(missing, 401, "unauthorized")
    assert missing.headers["www-authenticate"] == "Bearer"

    wrong = client.get("/predict", headers=auth())
    assert_error(wrong, 405, "method_not_allowed")
    assert wrong.headers["allow"] == "POST"


def test_both_auth_styles_and_bad_keys(client):
    payload = {"channel": "email", "subject": "Refund", "text": "I was charged twice"}
    assert client.post("/predict", json=payload, headers=auth()).status_code == 200
    bearer = client.post(
        "/predict",
        json=payload,
        headers={"authorization": "Bearer test-key"},
    )
    assert bearer.status_code == 200
    either = client.post(
        "/predict",
        json=payload,
        headers={"x-api-key": "nope", "authorization": "Bearer test-key"},
    )
    assert either.status_code == 200

    missing = assert_error(client.post("/predict", json=payload), 401, "unauthorized")
    assert "Missing API key" in missing["error"]["message"]
    invalid = assert_error(
        client.post("/predict", json=payload, headers=auth("nope")),
        401,
        "unauthorized",
    )
    assert invalid["error"]["message"] == "Invalid API key."


def test_unset_api_key_rejects_predict():
    payload = {"channel": "chat", "text": "hello"}
    with client_for(api_key=None, use_default_engine=True) as client:
        body = assert_error(client.post("/predict", json=payload), 401, "unauthorized")
        assert "Missing API key" in body["error"]["message"]
        assert client.get("/health").status_code == 200


def test_whitespace_key_is_not_stripped():
    payload = {"channel": "chat", "text": "hello"}
    with client_for(api_key="test key") as client:
        assert client.post("/predict", json=payload, headers=auth("test key")).status_code == 200
        assert client.post("/predict", json=payload, headers=auth("testkey")).status_code == 401
        assert client.post("/predict", json=payload, headers=auth(" test key")).status_code == 401


def test_check_order(client):
    # Wrong key wins over a bad content type, a huge body, and broken JSON.
    huge = b"x" * (PREDICT_MAX_BYTES + 1)
    wrong_key = client.post(
        "/predict",
        content=huge,
        headers={"content-type": "text/plain", "x-api-key": "nope"},
    )
    assert_error(wrong_key, 401, "unauthorized")

    bad_type = client.post(
        "/predict",
        content=b"{not json",
        headers={"content-type": "text/plain", "x-api-key": API_KEY},
    )
    assert_error(bad_type, 415, "unsupported_media_type")

    missing_type = client.post("/predict", content=b"{}", headers=auth())
    assert_error(missing_type, 415, "unsupported_media_type")

    too_big = client.post(
        "/predict",
        content=huge,
        headers={"content-type": "application/json", "x-api-key": API_KEY},
    )
    assert_error(too_big, 413, "payload_too_large")

    exact = client.post(
        "/predict",
        content=b"x" * PREDICT_MAX_BYTES,
        headers={"content-type": "application/json", "x-api-key": API_KEY},
    )
    assert_error(exact, 400, "malformed_json")

    broken = client.post(
        "/predict",
        content=b"{not json",
        headers={"content-type": "application/json; charset=utf-8", "x-api-key": API_KEY},
    )
    assert_error(broken, 400, "malformed_json")

    invalid = client.post(
        "/predict",
        json={"channel": "sms", "text": "hi"},
        headers=auth(),
    )
    body = assert_error(invalid, 422, "validation_error")
    assert body["error"]["details"][0]["field"] == "channel"
    assert "index" not in body["error"]["details"][0]


def test_request_id_is_echoed_on_errors_and_success(client):
    denied = client.post("/predict", json={"text": "hi"}, headers={"x-request-id": "req-1"})
    assert denied.status_code == 401
    assert denied.headers["x-request-id"] == "req-1"

    ok = client.post(
        "/predict",
        json={"channel": "chat", "text": "hi"},
        headers={**auth(), "x-request-id": "req-2"},
    )
    assert ok.status_code == 200
    assert ok.headers["x-request-id"] == "req-2"

    poisoned = client.get("/missing", headers={"x-request-id": "bad\nid"})
    assert poisoned.status_code == 404
    assert "x-request-id" not in poisoned.headers


def test_valid_edge_inputs_return_200(client):
    cases = [
        {"channel": "chat", "text": "🙏🙏"},
        {"channel": "chat", "text": "කාර් එකේ මගේ බෑග් එක අමතක වුණා"},
        {"channel": "email", "text": "hi", "language": "en", "extra": {"nested": True}},
        {"channel": "chat", "text": "a" * 10000},
        {"channel": "email", "subject": "s" * 500, "text": "body", "ticket_id": "T" * 64},
        {"channel": "chat", "text": "\u200bhi"},
    ]
    for payload in cases:
        response = client.post("/predict", json=payload, headers=auth())
        assert response.status_code == 200, payload
        body = response.json()
        assert_schema("predict_response", body)
        if "ticket_id" in payload:
            assert body["ticket_id"] == payload["ticket_id"]


def test_validation_messages(client):
    whitespace = client.post(
        "/predict",
        json={"channel": "chat", "text": " \n\t "},
        headers=auth(),
    )
    body = assert_error(whitespace, 422, "validation_error")
    assert body["error"]["message"] == "Request failed validation."
    assert (
        body["error"]["details"][0]["issue"] == "must contain at least one non-whitespace character"
    )

    too_long = client.post(
        "/predict",
        json={"channel": "chat", "text": "a" * 10001},
        headers=auth(),
    )
    assert_error(too_long, 422, "validation_error")

    subject = client.post(
        "/predict",
        json={"channel": "email", "subject": "s" * 501, "text": "hi"},
        headers=auth(),
    )
    details = assert_error(subject, 422, "validation_error")["error"]["details"]
    assert details[0]["field"] == "subject"

    null_subject = client.post(
        "/predict",
        json={"channel": "email", "subject": None, "text": "hi"},
        headers=auth(),
    )
    assert_error(null_subject, 422, "validation_error")

    array_body = client.post("/predict", json=["nope"], headers=auth())
    assert_error(array_body, 422, "validation_error")

    raw = client.post(
        "/predict",
        content=b"\xff\xfe",
        headers={"content-type": "application/json", "x-api-key": API_KEY},
    )
    assert_error(raw, 400, "malformed_json")


def test_batch_contract(client):
    tickets = [
        {"ticket_id": "b", "channel": "chat", "text": "short"},
        {
            "ticket_id": "a",
            "channel": "email",
            "subject": "Bill",
            "text": "charged twice please refund",
        },
    ]
    first = client.post("/predict/batch", json={"tickets": tickets}, headers=auth())
    second = client.post(
        "/predict/batch", json={"tickets": list(reversed(tickets))}, headers=auth()
    )
    assert first.status_code == 200, first.text
    assert_schema("batch_response", first.json())
    assert first.json()["meta"]["count"] == 2
    assert [row["ticket_id"] for row in first.json()["predictions"]] == ["b", "a"]
    by_first = {row["ticket_id"]: row for row in first.json()["predictions"]}
    by_second = {row["ticket_id"]: row for row in second.json()["predictions"]}
    assert by_first == by_second

    repeated = client.post("/predict/batch", json={"tickets": tickets}, headers=auth())
    assert [row["ticket_id"] for row in repeated.json()["predictions"]] == ["b", "a"]
    assert by_first == {row["ticket_id"]: row for row in repeated.json()["predictions"]}


def test_batch_is_atomic(client):
    tickets = []
    for index in range(5):
        tickets.append({"ticket_id": f"id{index}", "channel": "chat", "text": "hello"})
    tickets[1]["text"] = "   "
    del tickets[4]["ticket_id"]
    response = client.post("/predict/batch", json={"tickets": tickets}, headers=auth())
    body = assert_error(response, 422, "validation_error")
    indexes = {item["index"] for item in body["error"]["details"] if "index" in item}
    assert indexes == {1, 4}
    assert "predictions" not in body

    duplicate = client.post(
        "/predict/batch",
        json={
            "tickets": [
                {"ticket_id": "same", "channel": "chat", "text": "one"},
                {"ticket_id": "other", "channel": "chat", "text": "two"},
                {"ticket_id": "same", "channel": "chat", "text": "three"},
            ]
        },
        headers=auth(),
    )
    details = assert_error(duplicate, 422, "validation_error")["error"]["details"]
    assert details[-1]["index"] == 2
    assert details[-1]["issue"] == "duplicate of an earlier item"

    empty = client.post("/predict/batch", json={"tickets": []}, headers=auth())
    assert_error(empty, 422, "validation_error")
    too_many = {
        "tickets": [{"ticket_id": f"n{i}", "channel": "chat", "text": "hi"} for i in range(101)]
    }
    assert_error(
        client.post("/predict/batch", json=too_many, headers=auth()), 422, "validation_error"
    )

    hundred = {
        "tickets": [
            {"ticket_id": f"n{i}", "channel": "chat", "text": f"hi {i}"} for i in range(100)
        ]
    }
    ok = client.post("/predict/batch", json=hundred, headers=auth())
    assert ok.status_code == 200
    assert ok.json()["meta"]["count"] == 100


def test_model_loading_and_internal_error():
    payload = {"channel": "chat", "text": "hello"}
    with client_for(engine=None, use_default_engine=False) as client:
        health = client.get("/health")
        assert health.status_code == 503
        assert_schema("health_response", health.json())
        assert health.json()["status"] == "loading"
        assert health.json()["model_version"] is None

        loading = assert_error(
            client.post("/predict", json=payload, headers=auth()), 503, "model_loading"
        )
        assert loading["error"]["code"] == "model_loading"
        assert client.post("/predict", json=payload, headers=auth()).headers["retry-after"] == "2"
        # Invalid input is still 422 while the model is down.
        assert_error(
            client.post("/predict", json={"channel": "sms", "text": "hi"}, headers=auth()),
            422,
            "validation_error",
        )

    with client_for(engine=BoomEngine()) as client:
        failed = client.post("/predict", json=payload, headers=auth())
        assert_error(failed, 500, "internal_error")


def test_compare_digest_is_used(monkeypatch):
    called = {"count": 0}
    import app.auth as auth_module

    real = auth_module.hmac.compare_digest

    def wrapped(left, right):
        called["count"] += 1
        return real(left, right)

    monkeypatch.setattr(auth_module.hmac, "compare_digest", wrapped)
    with client_for() as client:
        client.post("/predict", json={"channel": "chat", "text": "hi"}, headers=auth())
    assert called["count"] >= 1


def test_chunked_and_declared_size_limits():
    captured = {"called": False}

    async def inner(scope, receive, send):
        captured["called"] = True
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": b"{}"})

    from app.config import Settings

    settings = Settings(api_key=API_KEY, predict_max_bytes=100, log_level="warning")
    app = ContractMiddleware(inner, settings)

    async def _run(headers, chunks):
        captured["called"] = False
        messages = []

        async def send(message):
            messages.append(message)

        pending = list(chunks)

        async def receive():
            if pending:
                body, more = pending.pop(0)
                return {"type": "http.request", "body": body, "more_body": more}
            return {"type": "http.disconnect"}

        scope = {"type": "http", "method": "POST", "path": "/predict", "headers": headers}
        await app(scope, receive, send)
        return messages

    base = [
        (b"content-type", b"application/json"),
        (b"x-api-key", b"test-key"),
    ]
    streamed = asyncio.run(_run(base, [(b"x" * 60, True), (b"x" * 50, False)]))
    assert streamed[0]["status"] == 413
    assert captured["called"] is False

    declared = asyncio.run(_run([*base, (b"content-length", b"10000")], [(b"{}", False)]))
    assert declared[0]["status"] == 413
    assert captured["called"] is False

    authed = asyncio.run(
        _run(
            [(b"content-type", b"text/plain"), (b"content-length", b"10000")],
            [(b"{}", False)],
        )
    )
    assert authed[0]["status"] == 401
    body = orjson.loads(authed[1]["body"])
    assert body["error"]["code"] == "unauthorized"
