"""Response dictionaries. Plain dicts rendered with orjson by the app."""

from __future__ import annotations

from typing import Any


def prediction_to_dict(pred: dict[str, Any]) -> dict[str, Any]:
    body: dict[str, Any] = {
        "category": pred["category"],
        "secondary_category": pred["secondary_category"],
        "team": pred["team"],
        "is_urgent": bool(pred["is_urgent"]),
        "confidence": float(pred["confidence"]),
        "model_version": str(pred["model_version"]),
    }
    if pred.get("ticket_id") is not None:
        body = {"ticket_id": pred["ticket_id"], **body}
    if "needs_human_review" in pred:
        body["needs_human_review"] = bool(pred["needs_human_review"])
    return body


def batch_to_dict(
    predictions: list[dict[str, Any]], model_version: str, elapsed_ms: float
) -> dict[str, Any]:
    rendered = [prediction_to_dict(pred) for pred in predictions]
    return {
        "predictions": rendered,
        "meta": {
            "count": len(rendered),
            "model_version": model_version,
            "processing_time_ms": int(elapsed_ms),
        },
    }
