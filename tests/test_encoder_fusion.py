"""Encoder fusion is optional. A failing encoder leaves the classical row in place."""

from __future__ import annotations

import numpy as np

from app.inference import Engine, fuse_ticket_probabilities
from app.labels import CATEGORIES, SECONDARY_CATEGORIES

FUSION = {
    "encoder_weight": {"category": 1.0, "secondary": 1.0, "urgent": 1.0},
    "temperature": {"category": 1.0, "secondary": 1.0, "urgent": 1.0},
}


class _Vectorizer:
    def transform(self, texts):
        return np.zeros((len(texts), 2))


class _Category:
    classes_ = list(CATEGORIES)

    def predict_proba(self, matrix):
        proba = np.zeros((matrix.shape[0], len(CATEGORIES)))
        proba[:, 0] = 1.0
        return proba


class _Urgent:
    classes_ = [0, 1]

    def predict_proba(self, matrix):
        return np.tile(np.array([[0.2, 0.8]]), (matrix.shape[0], 1))


class _SpamEncoder:
    def probabilities(self, texts):
        category = np.zeros((len(texts), len(CATEGORIES)))
        category[:, -1] = 1.0
        secondary = np.zeros((len(texts), len(SECONDARY_CATEGORIES)))
        urgent = np.zeros(len(texts))
        return category, secondary, urgent


class _BoomEncoder:
    def probabilities(self, texts):
        raise RuntimeError("onnx session failed")


def _engine() -> Engine:
    return Engine(
        {
            "model_version": "v0-test",
            "vectorizer": _Vectorizer(),
            "category_model": _Category(),
            "category_labels": list(CATEGORIES),
            "secondary_models": {},
            "secondary_labels": list(SECONDARY_CATEGORIES),
            "urgent_model": _Urgent(),
            "thresholds": {"secondary": 0.5, "urgent": 0.5, "review": 0.5},
        }
    )


def _ticket() -> dict:
    return {"channel": "chat", "subject": "hi", "text": "hello", "ticket_id": "t1"}


def test_unattached_engine_stays_classical():
    prediction = _engine().predict(_ticket())
    assert prediction["category"] == "payment_refund"
    assert prediction["is_urgent"] is True
    assert prediction["model_version"] == "v0-test"


def test_attached_encoder_is_fused_and_rules_still_run():
    engine = _engine()
    engine.attach_encoder(_SpamEncoder(), FUSION)
    prediction = engine.predict(_ticket())
    assert prediction["category"] == "spam_irrelevant"
    assert prediction["secondary_category"] is None
    assert prediction["is_urgent"] is False
    assert prediction["model_version"] == "v0-test"


def test_encoder_failure_keeps_the_classical_row():
    category = np.array([[0.2, 0.8]])
    secondary = np.zeros((1, 5))
    urgent = np.array([0.2])
    fused_category, fused_secondary, fused_urgent = fuse_ticket_probabilities(
        category,
        secondary,
        urgent,
        ["hello"],
        _BoomEncoder(),
        FUSION,
    )
    assert np.allclose(fused_category, category)
    assert np.allclose(fused_secondary, secondary)
    assert np.allclose(fused_urgent, urgent)


def test_one_ticket_can_fail_without_dropping_the_other():
    class _Flaky:
        def __init__(self) -> None:
            self.calls = 0

        def probabilities(self, texts):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("second ticket")
            category = np.zeros((1, 2))
            category[0, 0] = 1.0
            return category, np.zeros((1, 5)), np.zeros(1)

    category = np.array([[0.25, 0.75], [0.25, 0.75]])
    secondary = np.zeros((2, 5))
    urgent = np.array([0.1, 0.2])
    fused_category, _, fused_urgent = fuse_ticket_probabilities(
        category,
        secondary,
        urgent,
        ["a", "b"],
        _Flaky(),
        FUSION,
    )
    assert fused_category[0, 0] > 0.99
    assert np.allclose(fused_category[1], category[1])
    assert fused_urgent[1] == 0.2
