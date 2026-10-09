"""Pick secondary and urgency thresholds without looking at validation.

The grid is scored on probabilities from a model that did not train on those
rows. Category is argmax, so the grid does not change macro-F1. A point is
kept only when it gives up at most 0.01 secondary exact-match, and a tie
stays near 0.5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.inference import apply_rules
from app.labels import CATEGORIES, SECONDARY_CATEGORIES
from ml.evaluate import _binary_f1

GRID = (0.3, 0.4, 0.5, 0.6, 0.7)
REVIEW_THRESHOLD = 0.5


@dataclass(frozen=True)
class ThresholdPoint:
    secondary: float
    urgent: float
    secondary_exact: float
    urgent_f1: float


def decide(
    category_proba: np.ndarray,
    secondary_proba: np.ndarray,
    urgent_proba: np.ndarray,
    thresholds: dict[str, float],
) -> list[dict]:
    labels = list(CATEGORIES)
    secondary_labels = list(SECONDARY_CATEGORIES)
    return [
        apply_rules(
            category_proba[i],
            labels,
            secondary_proba[i],
            secondary_labels,
            float(urgent_proba[i]),
            thresholds,
        )
        for i in range(category_proba.shape[0])
    ]


def score_thresholds(
    category_proba: np.ndarray,
    secondary_proba: np.ndarray,
    urgent_proba: np.ndarray,
    gold_secondary: list[str | None],
    gold_urgent: list[bool],
    *,
    secondary: float,
    urgent: float,
) -> ThresholdPoint:
    thresholds = {"secondary": secondary, "urgent": urgent, "review": REVIEW_THRESHOLD}
    predictions = decide(category_proba, secondary_proba, urgent_proba, thresholds)
    pred_secondary = [item["secondary_category"] for item in predictions]
    pred_urgent = [bool(item["is_urgent"]) for item in predictions]
    matches = sum(gold == pred for gold, pred in zip(gold_secondary, pred_secondary, strict=True))
    exact = matches / len(gold_secondary) if gold_secondary else 0.0
    _precision, _recall, f1 = _binary_f1(gold_urgent, pred_urgent)
    return ThresholdPoint(secondary, urgent, exact, f1)


def select_thresholds(
    category_proba: np.ndarray,
    secondary_proba: np.ndarray,
    urgent_proba: np.ndarray,
    gold_secondary: list[str | None],
    gold_urgent: list[bool],
) -> tuple[dict[str, float], ThresholdPoint]:
    """Return `{secondary, urgent, review}` and the winning grid point."""
    points = [
        score_thresholds(
            category_proba,
            secondary_proba,
            urgent_proba,
            gold_secondary,
            gold_urgent,
            secondary=secondary,
            urgent=urgent,
        )
        for secondary in GRID
        for urgent in GRID
    ]
    best_secondary = max(point.secondary_exact for point in points)
    eligible = [point for point in points if point.secondary_exact >= best_secondary - 0.01]
    best_urgent = max(point.urgent_f1 for point in eligible)
    finalists = [point for point in eligible if point.urgent_f1 >= best_urgent - 0.005]
    chosen = min(
        finalists,
        key=lambda point: (
            abs(point.secondary - 0.5) + abs(point.urgent - 0.5),
            point.secondary,
            point.urgent,
        ),
    )
    return {
        "secondary": chosen.secondary,
        "urgent": chosen.urgent,
        "review": REVIEW_THRESHOLD,
    }, chosen
