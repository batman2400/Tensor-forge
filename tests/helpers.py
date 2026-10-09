"""Shared API-test helpers."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from jsonschema import Draft202012Validator
from starlette.testclient import TestClient

from app.config import JOB_MAX_BYTES, PREDICT_MAX_BYTES, Settings
from app.labels import TEAM_BY_CATEGORY
from app.main import create_app

ROOT = Path(__file__).resolve().parents[1]
API_KEY = "test-key"


class EchoEngine:
    """Deterministic stand-in. Output depends only on the ticket, not on batch position."""

    model_version = "v0-test"
    config_name = "echo"

    def predict(self, ticket: dict) -> dict:
        return self.predict_many([ticket])[0]

    def predict_many(self, tickets: list[dict]) -> list[dict]:
        predictions = []
        for ticket in tickets:
            category = {
                "email": "payment_refund",
                "chat": "general_inquiry",
                "call_transcript": "lost_item",
            }[ticket["channel"]]
            confidence = round(min(0.99, 0.25 + (len(ticket["text"]) % 50) / 100), 4)
            predictions.append(
                {
                    "category": category,
                    "secondary_category": None,
                    "team": TEAM_BY_CATEGORY[category],
                    "is_urgent": False,
                    "confidence": confidence,
                    "model_version": self.model_version,
                    "needs_human_review": confidence < 0.5,
                    "ticket_id": ticket.get("ticket_id"),
                }
            )
        return predictions


class BoomEngine:
    model_version = "v0-boom"
    config_name = "boom"

    def predict(self, ticket: dict) -> dict:
        raise RuntimeError("secret stack marker")

    def predict_many(self, tickets: list[dict]) -> list[dict]:
        raise RuntimeError("secret stack marker")


@lru_cache
def validator(name: str) -> Draft202012Validator:
    schema = json.loads((ROOT / "spec" / f"{name}.schema.json").read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


def assert_schema(name: str, instance: object) -> None:
    errors = sorted(validator(name).iter_errors(instance), key=lambda err: list(err.path))
    assert not errors, f"{name}: {errors[0].message} at {list(errors[0].path)}"


def assert_error(response, status: int, code: str) -> dict:
    assert response.status_code == status, response.text
    body = response.json()
    assert_schema("error_response", body)
    assert body["error"]["code"] == code
    lowered = response.text.lower()
    assert "traceback" not in lowered
    assert "secret stack marker" not in lowered
    return body


@contextmanager
def client_for(
    *,
    engine: object | None = None,
    api_key: str | None = API_KEY,
    predict_max_bytes: int = PREDICT_MAX_BYTES,
    job_max_bytes: int = JOB_MAX_BYTES,
    job_db_path: str | None = None,
    job_retention_seconds: int = 12 * 60 * 60,
    job_tombstone_seconds: int = 7 * 24 * 60 * 60,
    use_default_engine: bool = True,
) -> Iterator[TestClient]:
    if engine is None and use_default_engine:
        engine = EchoEngine()
    with tempfile.TemporaryDirectory(prefix="tf-jobs-") as directory:
        settings = Settings(
            api_key=api_key,
            log_level="warning",
            inference_threads=1,
            predict_max_bytes=predict_max_bytes,
            job_max_bytes=job_max_bytes,
            job_db_path=job_db_path or str(Path(directory) / "jobs.sqlite3"),
            job_retention_seconds=job_retention_seconds,
            job_tombstone_seconds=job_tombstone_seconds,
        )
        app = create_app(settings=settings, engine=engine, autoload=False)
        with TestClient(app) as client:
            yield client


def auth(key: str = API_KEY) -> dict[str, str]:
    return {"x-api-key": key}
