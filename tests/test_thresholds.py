"""Threshold grid: it may move urgency, and it must not move category."""

from __future__ import annotations

import numpy as np

from app.labels import CATEGORIES, SECONDARY_CATEGORIES
from ml.thresholds import decide, select_thresholds


def _category_proba(label: str, count: int) -> np.ndarray:
    proba = np.zeros((count, len(CATEGORIES)), dtype=np.float64)
    proba[:, CATEGORIES.index(label)] = 1.0
    return proba


def test_higher_urgent_threshold_wins_when_it_removes_false_alarms():
    count = 4
    category = _category_proba("payment_refund", count)
    secondary = np.zeros((count, len(SECONDARY_CATEGORIES)), dtype=np.float64)
    urgent = np.array([0.85, 0.85, 0.65, 0.65], dtype=np.float64)
    gold_urgent = [True, True, False, False]
    gold_secondary = [None, None, None, None]
    thresholds, point = select_thresholds(category, secondary, urgent, gold_secondary, gold_urgent)
    assert thresholds["urgent"] == 0.7
    assert thresholds["review"] == 0.5
    assert point.urgent_f1 == 1.0
    predictions = decide(category, secondary, urgent, thresholds)
    assert [item["category"] for item in predictions] == ["payment_refund"] * count
    assert [item["is_urgent"] for item in predictions] == [True, True, False, False]


def test_tie_stays_at_one_half():
    count = 4
    category = _category_proba("payment_refund", count)
    secondary = np.zeros((count, len(SECONDARY_CATEGORIES)), dtype=np.float64)
    urgent = np.full(count, 0.2, dtype=np.float64)
    thresholds, _point = select_thresholds(
        category,
        secondary,
        urgent,
        [None] * count,
        [False] * count,
    )
    assert thresholds["secondary"] == 0.5
    assert thresholds["urgent"] == 0.5
