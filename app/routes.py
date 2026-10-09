"""HTTP handlers for `/`, `/health`, `/predict`, `/predict/batch`, and `/batch/jobs`."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from typing import Any
from urllib.parse import parse_qsl

import orjson
from starlette.requests import Request
from starlette.responses import Response

from app.errors import ApiError, OrjsonResponse, error_body
from app.schemas import batch_to_dict, prediction_to_dict
from app.validation import validate_batch, validate_single

logger = logging.getLogger("tensorforge")

BATCH_MAX_ITEMS = 100
JOB_MAX_ITEMS = 5000
IDEMPOTENCY_MAX_LENGTH = 128
RESULTS_LIMIT_MAX = 5000
_DIGITS = re.compile(r"[0-9]+$")


def _ticket(obj: dict[str, Any]) -> dict[str, Any]:
    ticket: dict[str, Any] = {
        "channel": obj["channel"],
        "subject": obj.get("subject") or "",
        "text": obj["text"],
    }
    if "ticket_id" in obj:
        ticket["ticket_id"] = obj["ticket_id"]
    return ticket


def _require_engine(runtime: Any) -> Any:
    engine = runtime.engine
    if engine is None:
        raise ApiError(
            503,
            "model_loading",
            "Model is loading.",
            headers={"Retry-After": "2"},
        )
    return engine


def _idempotency_key(request: Request) -> str | None:
    raw = request.headers.get("idempotency-key")
    if raw is None or raw == "":
        return None
    return raw


def _bounded_int(value: str, *, maximum: int) -> int | None:
    if _DIGITS.fullmatch(value) is None or len(value) > 10:
        return None
    parsed = int(value)
    if parsed > maximum:
        return None
    return parsed


def _parse_results_query(request: Request) -> tuple[int, int | None]:
    """Read `offset` and `limit` ourselves so a bad value stays a contract 422."""
    raw = request.scope.get("query_string", b"")
    text = raw.decode("latin-1") if isinstance(raw, bytes) else str(raw)
    errors: list[dict[str, Any]] = []
    offset = 0
    limit: int | None = None
    saw_offset = False
    saw_limit = False
    for key, value in parse_qsl(text, keep_blank_values=True):
        if key == "offset":
            parsed = _bounded_int(value, maximum=1_000_000_000)
            if saw_offset or parsed is None:
                errors.append({"field": "offset", "issue": "must be an integer >= 0"})
            else:
                offset = parsed
            saw_offset = True
        elif key == "limit":
            parsed = _bounded_int(value, maximum=RESULTS_LIMIT_MAX)
            if saw_limit or parsed is None or parsed < 1:
                errors.append({"field": "limit", "issue": "must be an integer between 1 and 5000"})
            else:
                limit = parsed
            saw_limit = True
    if errors:
        raise ApiError(422, "validation_error", "Request failed validation.", errors)
    return offset, limit


def _job_http(view: Any) -> OrjsonResponse:
    if view.kind == "missing":
        raise ApiError(404, "job_not_found", "No job with this id.")
    if view.kind == "expired":
        raise ApiError(410, "job_expired", "Job results have expired.")
    if view.kind == "not_ready":
        raise ApiError(
            409,
            "job_not_ready",
            f"Job status is '{view.status}'. Results are available once it is 'succeeded'.",
        )
    headers = None
    if view.body["status"] in {"queued", "running"}:
        headers = {"Retry-After": "2"}
    return OrjsonResponse(view.body, headers=headers)


def register_routes(api: Any, runtime: Any) -> None:
    @api.get("/", response_model=None)
    async def root() -> OrjsonResponse:
        engine = runtime.engine
        return OrjsonResponse(
            {
                "service": "tensorforge",
                "model_version": None if engine is None else engine.model_version,
                "model_loaded": engine is not None,
                "endpoints": ["/health", "/predict", "/predict/batch", "/batch/jobs", "/demo/"],
            }
        )

    @api.api_route("/health", methods=["GET", "HEAD"], response_model=None)
    async def health() -> OrjsonResponse:
        engine = runtime.engine
        if engine is None:
            return OrjsonResponse(
                {"status": "loading", "model_version": None, "model_loaded": False},
                status_code=503,
                headers={"Retry-After": "2"},
            )
        return OrjsonResponse(
            {"status": "ok", "model_version": engine.model_version, "model_loaded": True}
        )

    @api.post("/predict", response_model=None)
    async def predict(request: Request) -> OrjsonResponse:
        payload = request.scope.get("tf_payload")
        errors = validate_single(payload)
        if errors:
            raise ApiError(422, "validation_error", "Request failed validation.", errors)
        engine = _require_engine(runtime)
        request.scope["tf_tickets"] = 1
        loop = asyncio.get_running_loop()
        try:
            pred = await loop.run_in_executor(runtime.executor, engine.predict, _ticket(payload))
        except Exception:
            logger.exception("predict failed")
            return OrjsonResponse(
                error_body("internal_error", "Internal server error."), status_code=500
            )
        return OrjsonResponse(prediction_to_dict(pred))

    @api.post("/predict/batch", response_model=None)
    async def predict_batch(request: Request) -> OrjsonResponse:
        payload = request.scope.get("tf_payload")
        errors = validate_batch(payload, max_items=BATCH_MAX_ITEMS)
        if errors:
            raise ApiError(422, "validation_error", "Request failed validation.", errors)
        engine = _require_engine(runtime)
        tickets = [_ticket(item) for item in payload["tickets"]]
        request.scope["tf_tickets"] = len(tickets)
        loop = asyncio.get_running_loop()
        started = time.perf_counter()
        try:
            preds = await loop.run_in_executor(runtime.executor, engine.predict_many, tickets)
        except Exception:
            logger.exception("batch predict failed")
            return OrjsonResponse(
                error_body("internal_error", "Internal server error."), status_code=500
            )
        elapsed_ms = (time.perf_counter() - started) * 1000
        return OrjsonResponse(batch_to_dict(preds, engine.model_version, elapsed_ms))

    @api.post("/batch/jobs", response_model=None)
    async def submit_job(request: Request) -> OrjsonResponse:
        payload = request.scope.get("tf_payload")
        errors = validate_batch(payload, max_items=JOB_MAX_ITEMS)
        if errors:
            raise ApiError(422, "validation_error", "Request failed validation.", errors)
        key = _idempotency_key(request)
        if key is not None and len(key) > IDEMPOTENCY_MAX_LENGTH:
            raise ApiError(
                422,
                "validation_error",
                "Request failed validation.",
                [{"field": "Idempotency-Key", "issue": "must be at most 128 characters"}],
            )
        engine = _require_engine(runtime)
        tickets = [_ticket(item) for item in payload["tickets"]]
        request.scope["tf_tickets"] = len(tickets)
        digest = hashlib.sha256(orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)).hexdigest()
        outcome = runtime.jobs.submit(
            tickets,
            idempotency_key=key,
            payload_hash=digest,
            model_version=str(engine.model_version),
        )
        if outcome is None:
            raise ApiError(
                429,
                "too_many_jobs",
                "Job queue is full. Retry later.",
                headers={"Retry-After": "2"},
            )
        job_id = outcome.body["job_id"]
        return OrjsonResponse(
            outcome.body,
            status_code=202,
            headers={"Location": f"/batch/jobs/{job_id}", "Retry-After": "2"},
        )

    @api.get("/batch/jobs/{job_id}", response_model=None)
    async def get_job(job_id: str) -> OrjsonResponse:
        if len(job_id) > IDEMPOTENCY_MAX_LENGTH:
            raise ApiError(404, "job_not_found", "No job with this id.")
        return _job_http(runtime.jobs.get(job_id))

    @api.delete("/batch/jobs/{job_id}", response_model=None)
    async def delete_job(job_id: str) -> Response:
        if len(job_id) > IDEMPOTENCY_MAX_LENGTH or not runtime.jobs.delete(job_id):
            raise ApiError(404, "job_not_found", "No job with this id.")
        return Response(status_code=204)

    @api.get("/batch/jobs/{job_id}/results", response_model=None)
    async def job_results(job_id: str, request: Request) -> OrjsonResponse:
        offset, limit = _parse_results_query(request)
        if len(job_id) > IDEMPOTENCY_MAX_LENGTH:
            raise ApiError(404, "job_not_found", "No job with this id.")
        view = runtime.jobs.results(job_id, offset, limit)
        if view.kind == "ok":
            return OrjsonResponse(view.body)
        if view.kind == "not_ready":
            raise ApiError(
                409,
                "job_not_ready",
                f"Job status is '{view.status}'. Results are available once it is 'succeeded'.",
            )
        return _job_http(view)
