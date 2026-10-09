"""Job submit, poll, results, delete, capacity, idempotency, and restart."""

from __future__ import annotations

import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tests.helpers import EchoEngine, assert_error, assert_schema, auth, client_for

INTERRUPTED = "Service restarted while the job was running."


def envelope(*texts: str, channel: str = "chat") -> dict:
    return {
        "tickets": [
            {"ticket_id": f"T{index}", "channel": channel, "text": text}
            for index, text in enumerate(texts, start=1)
        ]
    }


def bulk(count: int) -> dict:
    return {
        "tickets": [
            {"ticket_id": f"T{index}", "channel": "chat", "text": "hello"} for index in range(count)
        ]
    }


class GateEngine:
    """Blocks inside the first predict so tests can observe a running job."""

    model_version = "v0-test"

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def predict(self, ticket: dict) -> dict:
        if not self.entered.is_set():
            self.entered.set()
            self.release.wait(timeout=30)
        return EchoEngine().predict(ticket)


class PacedEngine:
    model_version = "v0-test"

    def __init__(self) -> None:
        self.started = threading.Event()

    def predict(self, ticket: dict) -> dict:
        self.started.set()
        time.sleep(0.005)
        return EchoEngine().predict(ticket)


class OnceBoom:
    model_version = "v0-test"

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, ticket: dict) -> dict:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("secret stack marker")
        return EchoEngine().predict(ticket)


class PredictOnly:
    model_version = "v0-test"

    def predict(self, ticket: dict) -> dict:
        return EchoEngine().predict(ticket)

    def predict_many(self, tickets: list[dict]) -> list[dict]:
        raise AssertionError("jobs must call predict")


def wait_for(client, job_id: str, status: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        response = client.get(f"/batch/jobs/{job_id}", headers=auth())
        assert response.status_code == 200, response.text
        last = response.json()
        if last["status"] == status:
            return last
        time.sleep(0.01)
    raise AssertionError(last)


def test_submit_returns_202_with_location_and_retry_after(client):
    response = client.post(
        "/batch/jobs",
        json=envelope("hello there"),
        headers={**auth(), "X-Request-ID": "job-1"},
    )
    assert response.status_code == 202, response.text
    assert response.headers["content-type"].startswith("application/json")
    assert response.headers["x-request-id"] == "job-1"
    body = response.json()
    assert_schema("batch_job_status", body)
    uuid.UUID(body["job_id"])
    assert response.headers["location"] == f"/batch/jobs/{body['job_id']}"
    assert response.headers["retry-after"] == "2"
    assert body["status"] == "queued"
    assert body["total"] == 1
    assert body["processed"] == 0
    assert body["error"] is None
    assert body["started_at"] is None
    assert body["created_at"].endswith("Z")
    assert body["model_version"] == client.get("/health").json()["model_version"]

    done = wait_for(client, body["job_id"], "succeeded")
    assert_schema("batch_job_status", done)
    assert done["processed"] == 1
    assert done["finished_at"].endswith("Z")
    assert done["expires_at"].endswith("Z")
    finished = client.get(f"/batch/jobs/{body['job_id']}", headers=auth())
    assert "retry-after" not in finished.headers


def test_invalid_item_creates_no_job(client):
    response = client.post(
        "/batch/jobs",
        json={
            "tickets": [
                {"ticket_id": "ok", "channel": "chat", "text": "hello"},
                {"ticket_id": "bad", "channel": "chat"},
            ]
        },
        headers=auth(),
    )
    body = assert_error(response, 422, "validation_error")
    assert any(item.get("index") == 1 for item in body["error"]["details"])
    duplicate = client.post(
        "/batch/jobs",
        json={
            "tickets": [
                {"ticket_id": "same", "channel": "chat", "text": "one"},
                {"ticket_id": "same", "channel": "email", "text": "two"},
            ]
        },
        headers=auth(),
    )
    dup_body = assert_error(duplicate, 422, "validation_error")
    assert any(
        item["issue"] == "duplicate of an earlier item" for item in dup_body["error"]["details"]
    )
    assert client.app.state.runtime.jobs.count_rows() == 0


def test_validation_runs_before_model_loading():
    with client_for(engine=None, use_default_engine=False) as client:
        invalid = client.post(
            "/batch/jobs",
            json={"tickets": [{"ticket_id": "T1", "channel": "chat"}]},
            headers=auth(),
        )
        assert_error(invalid, 422, "validation_error")
        loading = client.post("/batch/jobs", json=envelope("hello"), headers=auth())
        assert_error(loading, 503, "model_loading")
        assert loading.headers["retry-after"] == "2"
        assert client.app.state.runtime.jobs.count_rows() == 0


def test_processed_is_monotonic():
    engine = PacedEngine()
    total = 55
    with client_for(engine=engine) as client:
        response = client.post("/batch/jobs", json=bulk(total), headers=auth())
        assert response.status_code == 202, response.text
        job_id = response.json()["job_id"]
        assert engine.started.wait(2)
        seen: list[int] = []
        deadline = time.monotonic() + 5
        status = None
        while time.monotonic() < deadline:
            body = client.get(f"/batch/jobs/{job_id}", headers=auth()).json()
            seen.append(body["processed"])
            assert body["processed"] <= body["total"]
            status = body["status"]
            if status == "succeeded":
                break
            time.sleep(0.005)
        assert status == "succeeded"
        assert seen[0] == 0 or 0 in seen
        assert seen[-1] == total
        assert all(seen[index] <= seen[index + 1] for index in range(len(seen) - 1))


def test_results_page_and_are_byte_identical(client):
    response = client.post("/batch/jobs", json=envelope("a", "b", "c", "d", "e"), headers=auth())
    job_id = response.json()["job_id"]
    wait_for(client, job_id, "succeeded")

    full = client.get(f"/batch/jobs/{job_id}/results", headers=auth())
    assert full.status_code == 200, full.text
    full_body = full.json()
    assert_schema("batch_job_results", full_body)
    assert full_body["limit"] == 5
    assert full_body["next_offset"] is None
    assert [item["ticket_id"] for item in full_body["predictions"]] == [
        "T1",
        "T2",
        "T3",
        "T4",
        "T5",
    ]
    again = client.get(f"/batch/jobs/{job_id}/results", headers=auth())
    assert full.content == again.content

    pages = []
    offset = 0
    while True:
        page = client.get(
            f"/batch/jobs/{job_id}/results",
            params={"offset": offset, "limit": 2},
            headers=auth(),
        )
        assert page.status_code == 200, page.text
        body = page.json()
        assert_schema("batch_job_results", body)
        assert body["offset"] == offset
        assert body["limit"] == 2
        pages.extend(body["predictions"])
        twin = client.get(
            f"/batch/jobs/{job_id}/results",
            params={"offset": offset, "limit": 2},
            headers=auth(),
        )
        assert page.content == twin.content
        if body["next_offset"] is None:
            break
        offset = body["next_offset"]
    assert [item["ticket_id"] for item in pages] == ["T1", "T2", "T3", "T4", "T5"]


def test_results_409_until_succeeded():
    gate = GateEngine()
    with client_for(engine=gate) as client:
        response = client.post("/batch/jobs", json=envelope("hold"), headers=auth())
        job_id = response.json()["job_id"]
        assert gate.entered.wait(2)
        denied = client.get(f"/batch/jobs/{job_id}/results", headers=auth())
        body = assert_error(denied, 409, "job_not_ready")
        assert body["error"]["message"] == (
            "Job status is 'running'. Results are available once it is 'succeeded'."
        )
        gate.release.set()
        wait_for(client, job_id, "succeeded")
        ready = client.get(f"/batch/jobs/{job_id}/results", headers=auth())
        assert ready.status_code == 200


def test_unknown_job_is_404(client):
    missing = client.get(f"/batch/jobs/{uuid.uuid4()}", headers=auth())
    body = assert_error(missing, 404, "job_not_found")
    assert body["error"]["message"] == "No job with this id."
    results = client.get(f"/batch/jobs/{uuid.uuid4()}/results", headers=auth())
    assert_error(results, 404, "job_not_found")
    deleted = client.delete(f"/batch/jobs/{uuid.uuid4()}", headers=auth())
    assert_error(deleted, 404, "job_not_found")


def test_delete_then_404(client):
    response = client.post("/batch/jobs", json=envelope("bye"), headers=auth())
    job_id = response.json()["job_id"]
    wait_for(client, job_id, "succeeded")
    deleted = client.delete(f"/batch/jobs/{job_id}", headers=auth())
    assert deleted.status_code == 204
    assert deleted.content == b""
    assert_error(client.get(f"/batch/jobs/{job_id}", headers=auth()), 404, "job_not_found")
    assert_error(client.get(f"/batch/jobs/{job_id}/results", headers=auth()), 404, "job_not_found")
    assert_error(client.delete(f"/batch/jobs/{job_id}", headers=auth()), 404, "job_not_found")


def test_delete_running_job_and_the_next_one_succeeds():
    gate = GateEngine()
    with client_for(engine=gate) as client:
        first = client.post("/batch/jobs", json=envelope("first"), headers=auth())
        first_id = first.json()["job_id"]
        assert gate.entered.wait(2)
        deleted = client.delete(f"/batch/jobs/{first_id}", headers=auth())
        assert deleted.status_code == 204
        assert_error(client.get(f"/batch/jobs/{first_id}", headers=auth()), 404, "job_not_found")
        second = client.post("/batch/jobs", json=envelope("second"), headers=auth())
        second_id = second.json()["job_id"]
        gate.release.set()
        done = wait_for(client, second_id, "succeeded")
        assert done["processed"] == 1


def test_expired_job_is_410_and_delete_removes_the_tombstone():
    with client_for(job_retention_seconds=0) as client:
        response = client.post(
            "/batch/jobs",
            json=envelope("expire me"),
            headers={**auth(), "Idempotency-Key": "again"},
        )
        job_id = response.json()["job_id"]
        deadline = time.monotonic() + 5
        status = None
        while time.monotonic() < deadline:
            status = client.get(f"/batch/jobs/{job_id}", headers=auth())
            if status.status_code == 410:
                break
            time.sleep(0.01)
        body = assert_error(status, 410, "job_expired")
        assert body["error"]["message"] == "Job results have expired."
        results = client.get(f"/batch/jobs/{job_id}/results", headers=auth())
        assert_error(results, 410, "job_expired")
        replay = client.post(
            "/batch/jobs",
            json=envelope("new work"),
            headers={**auth(), "Idempotency-Key": "again"},
        )
        assert replay.status_code == 202
        assert replay.json()["job_id"] != job_id
        removed = client.delete(f"/batch/jobs/{job_id}", headers=auth())
        assert removed.status_code == 204
        assert_error(client.get(f"/batch/jobs/{job_id}", headers=auth()), 404, "job_not_found")


def test_idempotency_returns_the_original_job(client):
    headers = {**auth(), "Idempotency-Key": "stable"}
    first = client.post("/batch/jobs", json=envelope("one"), headers=headers)
    second = client.post("/batch/jobs", json=envelope("one", "two"), headers=headers)
    assert first.status_code == 202
    assert second.status_code == 202
    assert second.headers["location"] == first.headers["location"]
    assert second.json()["job_id"] == first.json()["job_id"]
    assert second.json()["total"] == 1
    fresh = client.post("/batch/jobs", json=envelope("other"), headers=auth())
    assert fresh.json()["job_id"] != first.json()["job_id"]


def test_fifth_active_job_is_429():
    gate = GateEngine()
    with client_for(engine=gate) as client:
        first = client.post(
            "/batch/jobs",
            json=envelope("hold"),
            headers={**auth(), "Idempotency-Key": "hold"},
        )
        assert first.status_code == 202, first.text
        job_id = first.json()["job_id"]
        assert gate.entered.wait(2)
        queued = []
        for index in range(3):
            response = client.post("/batch/jobs", json=envelope(f"q{index}"), headers=auth())
            assert response.status_code == 202, response.text
            queued.append(response.json()["job_id"])
        extra = client.post("/batch/jobs", json=envelope("overflow"), headers=auth())
        body = assert_error(extra, 429, "too_many_jobs")
        assert extra.headers["retry-after"] == "2"
        assert body["error"]["message"] == "Job queue is full. Retry later."
        replay = client.post(
            "/batch/jobs",
            json=envelope("different", "payload"),
            headers={**auth(), "Idempotency-Key": "hold"},
        )
        assert replay.status_code == 202
        assert replay.json()["job_id"] == job_id
        assert client.get(f"/batch/jobs/{job_id}", headers=auth()).json()["status"] == "running"
        for queued_id in queued:
            assert (
                client.get(f"/batch/jobs/{queued_id}", headers=auth()).json()["status"] == "queued"
            )
        waiting = client.get(f"/batch/jobs/{queued[0]}/results", headers=auth())
        assert "queued" in assert_error(waiting, 409, "job_not_ready")["error"]["message"]
        gate.release.set()


def test_parallel_idempotency_is_one_job(client):
    jobs = client.app.state.runtime.jobs
    tickets = [{"ticket_id": "T1", "channel": "chat", "subject": "", "text": "hello"}]

    def once():
        return jobs.submit(
            tickets,
            idempotency_key="race-key",
            payload_hash="same",
            model_version="v0-test",
        )

    with ThreadPoolExecutor(max_workers=20) as pool:
        outcomes = list(pool.map(lambda _: once(), range(20)))
    assert all(item is not None for item in outcomes)
    assert len({item.body["job_id"] for item in outcomes}) == 1
    wait_for(client, outcomes[0].body["job_id"], "succeeded")


def test_restart_marks_the_running_job_interrupted():
    gate = GateEngine()
    try:
        with tempfile.TemporaryDirectory() as directory:
            database = str(Path(directory) / "jobs.sqlite3")
            with client_for(engine=gate, job_db_path=database) as client:
                response = client.post("/batch/jobs", json=envelope("in flight"), headers=auth())
                job_id = response.json()["job_id"]
                assert gate.entered.wait(2)
                assert (
                    client.get(f"/batch/jobs/{job_id}", headers=auth()).json()["status"]
                    == "running"
                )
            with client_for(engine=EchoEngine(), job_db_path=database) as client:
                response = client.get(f"/batch/jobs/{job_id}", headers=auth())
                assert response.status_code == 200, response.text
                body = response.json()
                assert body["status"] == "failed"
                assert body["error"] == {"code": "interrupted", "message": INTERRUPTED}
                assert body["finished_at"] is not None
                assert body["job_id"] == job_id
                results = client.get(f"/batch/jobs/{job_id}/results", headers=auth())
                assert_error(results, 409, "job_not_ready")
    finally:
        gate.release.set()


def test_health_and_poll_stay_fast_while_a_large_job_is_running():
    gate = GateEngine()
    with client_for(engine=gate) as client:
        response = client.post("/batch/jobs", json=bulk(5000), headers=auth())
        assert response.status_code == 202, response.text
        job_id = response.json()["job_id"]
        assert gate.entered.wait(2)
        started = time.perf_counter()
        health = client.get("/health")
        assert health.status_code == 200
        assert time.perf_counter() - started < 1
        started = time.perf_counter()
        poll = client.get(f"/batch/jobs/{job_id}", headers=auth())
        assert poll.status_code == 200
        assert poll.json()["status"] == "running"
        assert poll.json()["total"] == 5000
        assert time.perf_counter() - started < 1
        assert poll.headers["retry-after"] == "2"
        started = time.perf_counter()
        live = client.post(
            "/predict",
            json={"channel": "chat", "text": "hi"},
            headers=auth(),
        )
        assert live.status_code == 200
        assert time.perf_counter() - started < 1
        assert client.delete(f"/batch/jobs/{job_id}", headers=auth()).status_code == 204
        gate.release.set()


def test_results_match_across_runs():
    payload = envelope("alpha", "beta mail", "gamma")

    def predictions() -> list[dict]:
        with client_for() as client:
            response = client.post("/batch/jobs", json=payload, headers=auth())
            job_id = response.json()["job_id"]
            wait_for(client, job_id, "succeeded")
            body = client.get(f"/batch/jobs/{job_id}/results", headers=auth()).json()
            return body["predictions"]

    assert predictions() == predictions()


def test_worker_survives_a_predict_error():
    with client_for(engine=OnceBoom()) as client:
        failed = client.post("/batch/jobs", json=envelope("boom"), headers=auth())
        failed_id = failed.json()["job_id"]
        body = wait_for(client, failed_id, "failed")
        assert body["error"]["code"] == "internal_error"
        assert body["error"]["message"] == "Internal server error."
        listed = client.get(f"/batch/jobs/{failed_id}", headers=auth())
        assert "secret stack marker" not in listed.text
        results = client.get(f"/batch/jobs/{failed_id}/results", headers=auth())
        assert_error(results, 409, "job_not_ready")
        assert "failed" in results.json()["error"]["message"]
        nxt = client.post("/batch/jobs", json=envelope("ok"), headers=auth())
        done = wait_for(client, nxt.json()["job_id"], "succeeded")
        assert done["processed"] == 1


def test_jobs_call_predict_not_predict_many():
    with client_for(engine=PredictOnly()) as client:
        response = client.post("/batch/jobs", json=envelope("one", "two"), headers=auth())
        job_id = response.json()["job_id"]
        wait_for(client, job_id, "succeeded")
        body = client.get(f"/batch/jobs/{job_id}/results", headers=auth()).json()
        assert [item["ticket_id"] for item in body["predictions"]] == ["T1", "T2"]


def test_job_routes_keep_the_check_order(client):
    missing = client.post(
        "/batch/jobs",
        content=b"nope",
        headers={"content-type": "text/plain"},
    )
    assert_error(missing, 401, "unauthorized")
    wrong_type = client.post(
        "/batch/jobs",
        content=b"nope",
        headers={**auth(), "content-type": "text/plain"},
    )
    assert_error(wrong_type, 415, "unsupported_media_type")
    broken = client.post(
        "/batch/jobs",
        content=b"{",
        headers={**auth(), "content-type": "application/json"},
    )
    assert_error(broken, 400, "malformed_json")
    no_key = client.get("/batch/jobs")
    assert_error(no_key, 401, "unauthorized")
    wrong_method = client.get("/batch/jobs", headers=auth())
    assert_error(wrong_method, 405, "method_not_allowed")
    assert wrong_method.headers["allow"] == "POST"
    slash = client.get("/batch/jobs/")
    assert_error(slash, 404, "not_found")
    long_key = client.post(
        "/batch/jobs",
        json=envelope("hello"),
        headers={**auth(), "Idempotency-Key": "k" * 129},
    )
    assert_error(long_key, 422, "validation_error")
    bad_page = client.get("/batch/jobs/whatever/results", params={"limit": "0"}, headers=auth())
    page_body = assert_error(bad_page, 422, "validation_error")
    assert page_body["error"]["details"][0]["field"] == "limit"
    assert "detail" not in page_body


def test_oversized_job_body_is_413():
    with client_for(job_max_bytes=80) as client:
        response = client.post("/batch/jobs", json=envelope("x" * 200), headers=auth())
        assert_error(response, 413, "payload_too_large")
        assert client.app.state.runtime.jobs.count_rows() == 0


def test_root_lists_jobs(client):
    assert "/batch/jobs" in client.get("/").json()["endpoints"]
