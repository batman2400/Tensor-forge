from __future__ import annotations

import numpy as np

from app.inference import Engine, apply_rules, resolve_field_priority
from app.labels import CATEGORIES, SECONDARY_CATEGORIES, TEAM_BY_CATEGORY
from app.text import build_input

THRESHOLDS = {"secondary": 0.5, "urgent": 0.5, "review": 0.5}


def _proba(category: str, confidence: float = 0.8) -> np.ndarray:
    proba = np.full(len(CATEGORIES), (1.0 - confidence) / (len(CATEGORIES) - 1))
    proba[list(CATEGORIES).index(category)] = confidence
    return proba


def _secondary(high: str | None = None, value: float = 0.0) -> np.ndarray:
    proba = np.zeros(len(SECONDARY_CATEGORIES))
    if high is not None:
        proba[list(SECONDARY_CATEGORIES).index(high)] = value
    return proba


def test_spam_clears_urgency_and_secondary():
    pred = apply_rules(
        _proba("spam_irrelevant", 0.9),
        list(CATEGORIES),
        np.full(len(SECONDARY_CATEGORIES), 0.99),
        list(SECONDARY_CATEGORIES),
        0.99,
        THRESHOLDS,
    )
    assert pred["category"] == "spam_irrelevant"
    assert pred["secondary_category"] is None
    assert pred["is_urgent"] is False
    assert pred["team"] == "Auto-close / Spam Filter"


def test_never_urgent_categories_stay_quiet():
    for category in ("account_promo", "app_technical", "general_inquiry"):
        pred = apply_rules(
            _proba(category),
            list(CATEGORIES),
            _secondary(),
            list(SECONDARY_CATEGORIES),
            0.99,
            THRESHOLDS,
        )
        assert pred["is_urgent"] is False
        assert pred["team"] == TEAM_BY_CATEGORY[category]


def test_secondary_skips_the_primary_and_respects_threshold():
    pred = apply_rules(
        _proba("payment_refund"),
        list(CATEGORIES),
        _secondary("payment_refund", 0.99) + _secondary("ride_trip_issue", 0.8),
        list(SECONDARY_CATEGORIES),
        0.1,
        THRESHOLDS,
    )
    assert pred["secondary_category"] == "ride_trip_issue"
    assert pred["is_urgent"] is False

    low = apply_rules(
        _proba("payment_refund"),
        list(CATEGORIES),
        _secondary("ride_trip_issue", 0.4),
        list(SECONDARY_CATEGORIES),
        0.9,
        THRESHOLDS,
    )
    assert low["secondary_category"] is None
    assert low["is_urgent"] is True
    assert low["team"] == "Payments & Refunds"


def test_review_flag_uses_rounded_confidence():
    pred = apply_rules(
        _proba("lost_item", 0.4),
        list(CATEGORIES),
        _secondary(),
        list(SECONDARY_CATEGORIES),
        0.0,
        THRESHOLDS,
    )
    assert pred["confidence"] == 0.4
    assert pred["needs_human_review"] is True
    assert pred["confidence"] <= 1


def _pred(category: str, confidence: float = 0.8) -> dict:
    pred = apply_rules(
        _proba(category, confidence),
        list(CATEGORIES),
        _secondary(),
        list(SECONDARY_CATEGORIES),
        0.1,
        THRESHOLDS,
    )
    return pred


def test_priority_keeps_combined_without_a_subject():
    combined = _pred("payment_refund", 0.7)
    assert (
        resolve_field_priority(combined, None, None, body_is_vague=False, subject_is_vague=True)
        is combined
    )


def test_priority_safety_on_either_field_wins():
    combined = _pred("ride_trip_issue", 0.95)
    body = _pred("payment_refund", 0.9)
    subject = _pred("safety_conduct", 0.88)
    chosen = resolve_field_priority(
        combined, body, subject, body_is_vague=False, subject_is_vague=False
    )
    assert chosen["category"] == "safety_conduct"
    assert chosen["team"] == "Trust & Safety"
    assert chosen["confidence"] == 0.88

    body_safety = _pred("safety_conduct", 0.91)
    subject_refund = _pred("payment_refund", 0.93)
    chosen = resolve_field_priority(
        combined, body_safety, subject_refund, body_is_vague=False, subject_is_vague=False
    )
    assert chosen is body_safety


def test_priority_vague_body_follows_the_subject():
    combined = _pred("ride_trip_issue", 0.6)
    body = _pred("general_inquiry", 0.33)
    subject = _pred("payment_refund", 0.84)
    chosen = resolve_field_priority(
        combined, body, subject, body_is_vague=True, subject_is_vague=False
    )
    assert chosen is subject


def test_priority_disagreement_keeps_the_body_and_its_review_flag():
    combined = _pred("ride_trip_issue", 0.95)
    sure_body = _pred("delivery_delay", 0.84)
    subject = _pred("payment_refund", 0.9)
    chosen = resolve_field_priority(
        combined, sure_body, subject, body_is_vague=False, subject_is_vague=False
    )
    assert chosen is sure_body
    assert chosen["needs_human_review"] is False

    unsure_body = _pred("delivery_delay", 0.42)
    chosen = resolve_field_priority(
        combined, unsure_body, subject, body_is_vague=False, subject_is_vague=False
    )
    assert chosen["category"] == "delivery_delay"
    assert chosen["confidence"] == 0.42
    assert chosen["needs_human_review"] is True


def test_priority_agreement_keeps_the_combined_score():
    combined = _pred("payment_refund", 0.61)
    body = _pred("payment_refund", 0.8)
    subject = _pred("payment_refund", 0.8)
    chosen = resolve_field_priority(
        combined, body, subject, body_is_vague=False, subject_is_vague=False
    )
    assert chosen is combined


class _Store:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def transform(self, texts):
        self.texts = list(texts)
        return np.zeros((len(texts), 1))


class _Scripted:
    classes_ = list(CATEGORIES)

    def __init__(self, store: _Store, table: dict[str, tuple[str, float]]) -> None:
        self.store = store
        self.table = table

    def predict_proba(self, matrix):
        proba = np.zeros((matrix.shape[0], len(CATEGORIES)))
        for row, text in enumerate(self.store.texts):
            label, confidence = self.table[text]
            proba[row, list(CATEGORIES).index(label)] = confidence
        return proba


class _Quiet:
    classes_ = [0, 1]

    def predict_proba(self, matrix):
        return np.tile(np.array([[1.0, 0.0]]), (matrix.shape[0], 1))


def _scripted(table: dict[str, tuple[str, float]]) -> Engine:
    store = _Store()
    return Engine(
        {
            "model_version": "v0-priority",
            "vectorizer": store,
            "category_model": _Scripted(store, table),
            "category_labels": list(CATEGORIES),
            "secondary_models": {},
            "secondary_labels": list(SECONDARY_CATEGORIES),
            "urgent_model": _Quiet(),
            "thresholds": THRESHOLDS,
        }
    )


def test_predict_uses_the_body_when_the_subject_names_another_issue():
    subject = "Charged twice"
    text = "The driver never arrived for the trip"
    table = {
        build_input("email", subject, text): ("spam_irrelevant", 0.99),
        build_input("email", None, text): ("ride_trip_issue", 0.86),
        build_input("email", None, subject): ("payment_refund", 0.9),
    }
    prediction = _scripted(table).predict(
        {"channel": "email", "subject": subject, "text": text, "ticket_id": "d1"}
    )
    assert prediction["category"] == "ride_trip_issue"
    assert prediction["team"] == "Ride Operations"
    assert prediction["confidence"] == 0.86
    assert prediction["needs_human_review"] is False
    assert prediction["ticket_id"] == "d1"


def test_predict_routes_safety_from_the_subject():
    subject = "Driver hit me"
    text = "Please refund last night's ride"
    table = {
        build_input("email", subject, text): ("payment_refund", 0.93),
        build_input("email", None, text): ("payment_refund", 0.9),
        build_input("email", None, subject): ("safety_conduct", 0.88),
    }
    prediction = _scripted(table).predict({"channel": "email", "subject": subject, "text": text})
    assert prediction["category"] == "safety_conduct"
    assert prediction["team"] == "Trust & Safety"
    assert prediction["confidence"] == 0.88


def test_predict_follows_a_specific_subject_when_the_body_is_vague():
    subject = "Charged twice"
    text = "please help"
    table = {
        build_input("email", subject, text): ("ride_trip_issue", 0.7),
        build_input("email", None, text): ("general_inquiry", 0.33),
        build_input("email", None, subject): ("payment_refund", 0.84),
    }
    prediction = _scripted(table).predict({"channel": "email", "subject": subject, "text": text})
    assert prediction["category"] == "payment_refund"
    assert prediction["confidence"] == 0.84
    assert prediction["needs_human_review"] is False


def test_predict_keeps_one_score_when_the_subject_is_empty_or_vague():
    text = "where is my order"
    combined = build_input("chat", "", text)
    engine = _scripted({combined: ("delivery_delay", 0.73)})
    prediction = engine.predict({"channel": "chat", "subject": None, "text": text})
    assert prediction["category"] == "delivery_delay"
    assert prediction["confidence"] == 0.73

    threaded = "The food was an hour late"
    combined_email = build_input("email", "Re:", threaded)
    engine = _scripted({combined_email: ("delivery_delay", 0.81)})
    prediction = engine.predict({"channel": "email", "subject": "Re:", "text": threaded})
    assert prediction["category"] == "delivery_delay"
    assert prediction["confidence"] == 0.81


def test_priority_ignores_a_vague_subject():
    combined = _pred("payment_refund", 0.77)
    body = _pred("delivery_delay", 0.8)
    subject = _pred("safety_conduct", 0.99)
    chosen = resolve_field_priority(
        combined, body, subject, body_is_vague=False, subject_is_vague=True
    )
    assert chosen is combined
