"""Metrics for category, secondary, urgency, calibration, and slices.

`append_experiment` writes one row to `ml/reports/experiments.csv`.
"""

from __future__ import annotations

import csv
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from app.labels import CATEGORIES, SECONDARY_CATEGORIES
from ml.data import TicketRow, load_all

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "ml" / "reports"
EXPERIMENTS_PATH = REPORTS_DIR / "experiments.csv"

CSV_COLUMNS = [
    "id",
    "date",
    "branch",
    "features",
    "hyperparameters",
    "stage_a_accuracy",
    "stage_a_macro_f1",
    "stage_a_urgent_f1",
    "stage_a_secondary_exact",
    "cv_accuracy",
    "cv_macro_f1",
    "cv_urgent_f1",
    "cv_secondary_exact",
    "notes",
]

METRIC_KEYS = (
    "n",
    "accuracy",
    "macro_f1",
    "per_class_f1",
    "per_class_support",
    "confusion_matrix",
    "secondary_exact",
    "secondary_per_class_f1",
    "urgent_precision",
    "urgent_recall",
    "urgent_f1",
    "ece",
    "brier",
    "urgent_brier",
    "by_language",
    "by_channel",
)


def _round(value: float) -> float:
    return round(float(value), 6)


def reliability_bins(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10
) -> list[dict[str, Any]]:
    """Bin predicted-class confidence against accuracy. Empty bins have n=0."""
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict[str, Any]] = []
    for index in range(n_bins):
        low, high = float(edges[index]), float(edges[index + 1])
        if index == n_bins - 1:
            mask = (confidence >= low) & (confidence <= high)
        else:
            mask = (confidence >= low) & (confidence < high)
        count = int(mask.sum())
        bins.append(
            {
                "low": low,
                "high": high,
                "n": count,
                "confidence": float(confidence[mask].mean()) if count else None,
                "accuracy": float(correct[mask].mean()) if count else None,
            }
        )
    return bins


def expected_calibration_error(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15
) -> float:
    """ECE of the predicted class: bin confidence against accuracy."""
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    total = len(confidence)
    if total == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for index in range(n_bins):
        low, high = edges[index], edges[index + 1]
        if index == n_bins - 1:
            mask = (confidence >= low) & (confidence <= high)
        else:
            mask = (confidence >= low) & (confidence < high)
        count = int(mask.sum())
        if count == 0:
            continue
        ece += (count / total) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(ece)


def multiclass_brier(proba: np.ndarray, gold: list[str], labels: list[str]) -> float:
    index = {label: i for i, label in enumerate(labels)}
    y_index = np.array([index[value] for value in gold])
    one_hot = np.zeros_like(proba, dtype=np.float64)
    one_hot[np.arange(len(gold)), y_index] = 1.0
    return float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))


def _binary_f1(gold: list[bool], pred: list[bool]) -> tuple[float, float, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        gold,
        pred,
        average="binary",
        pos_label=True,
        zero_division=0,
    )
    return _round(precision), _round(recall), _round(f1)


def _accuracy_f1(gold: list[str], pred: list[str], labels: list[str]) -> tuple[float, float]:
    if not gold:
        return 0.0, 0.0
    accuracy = accuracy_score(gold, pred)
    macro = f1_score(gold, pred, average="macro", labels=labels, zero_division=0)
    return _round(accuracy), _round(macro)


def _take(values: list[Any], indexes: list[int]) -> list[Any]:
    return [values[i] for i in indexes]


def compute_metrics(
    *,
    gold_category: list[str],
    pred_category: list[str],
    category_proba: np.ndarray,
    gold_secondary: list[str | None],
    pred_secondary: list[str | None],
    gold_urgent: list[bool],
    pred_urgent: list[bool],
    urgent_proba: np.ndarray,
    languages: list[str] | None = None,
    channels: list[str] | None = None,
    category_labels: list[str] | None = None,
) -> dict[str, Any]:
    """Full metric table. Probabilities are aligned to `category_labels`."""
    labels = list(category_labels or CATEGORIES)
    proba = np.asarray(category_proba, dtype=np.float64)
    confidence = proba.max(axis=1) if len(proba) else np.zeros(0)
    correct = np.array(
        [gold == pred for gold, pred in zip(gold_category, pred_category, strict=True)]
    )
    accuracy, macro_f1 = _accuracy_f1(gold_category, pred_category, labels)
    per_class = f1_score(gold_category, pred_category, average=None, labels=labels, zero_division=0)
    matrix = confusion_matrix(gold_category, pred_category, labels=labels)
    support = matrix.sum(axis=1)

    secondary_exact = 0.0
    if gold_secondary:
        matches = sum(
            gold == pred for gold, pred in zip(gold_secondary, pred_secondary, strict=True)
        )
        secondary_exact = matches / len(gold_secondary)

    secondary_f1: dict[str, float] = {}
    for label in SECONDARY_CATEGORIES:
        gold_bin = [value == label for value in gold_secondary]
        pred_bin = [value == label for value in pred_secondary]
        secondary_f1[label] = _binary_f1(gold_bin, pred_bin)[2]

    urgent_precision, urgent_recall, urgent_f1 = _binary_f1(gold_urgent, pred_urgent)
    urgent_scores = np.asarray(urgent_proba, dtype=np.float64)
    urgent_brier = float(np.mean((urgent_scores - np.asarray(gold_urgent, dtype=np.float64)) ** 2))

    by_language: dict[str, dict[str, float | int]] = {}
    if languages is not None:
        for language in sorted(set(languages)):
            indexes = [i for i, value in enumerate(languages) if value == language]
            slice_acc, slice_f1 = _accuracy_f1(
                _take(gold_category, indexes),
                _take(pred_category, indexes),
                labels,
            )
            by_language[language] = {"n": len(indexes), "accuracy": slice_acc, "macro_f1": slice_f1}

    by_channel: dict[str, dict[str, float | int]] = {}
    if channels is not None:
        for channel in sorted(set(channels)):
            indexes = [i for i, value in enumerate(channels) if value == channel]
            slice_acc, slice_f1 = _accuracy_f1(
                _take(gold_category, indexes),
                _take(pred_category, indexes),
                labels,
            )
            by_channel[channel] = {"n": len(indexes), "accuracy": slice_acc, "macro_f1": slice_f1}

    return {
        "n": len(gold_category),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_f1": {
            label: _round(score) for label, score in zip(labels, per_class, strict=True)
        },
        "per_class_support": {
            label: int(count) for label, count in zip(labels, support, strict=True)
        },
        "confusion_matrix": [[int(value) for value in row] for row in matrix.tolist()],
        "secondary_exact": _round(secondary_exact),
        "secondary_per_class_f1": secondary_f1,
        "urgent_precision": urgent_precision,
        "urgent_recall": urgent_recall,
        "urgent_f1": urgent_f1,
        "ece": _round(expected_calibration_error(confidence, correct.astype(np.float64))),
        "brier": _round(multiclass_brier(proba, gold_category, labels)),
        "urgent_brier": _round(urgent_brier),
        "by_language": by_language,
        "by_channel": by_channel,
    }


def majority_metrics(rows: list[TicketRow] | None = None) -> dict[str, Any]:
    """Train-set majority category, predicted for every validation ticket.

    Secondary is always null and urgency is always false. This is the floor.
    """
    rows = load_all() if rows is None else rows
    train = [row for row in rows if row.split == "train"]
    validation = [row for row in rows if row.split == "validation"]
    majority = Counter(row.category for row in train).most_common(1)[0][0]
    labels = list(CATEGORIES)
    column = labels.index(majority)
    proba = np.zeros((len(validation), len(labels)), dtype=np.float64)
    proba[:, column] = 1.0
    count = len(validation)
    metrics = compute_metrics(
        gold_category=[row.category for row in validation],
        pred_category=[majority] * count,
        category_proba=proba,
        gold_secondary=[row.secondary_category for row in validation],
        pred_secondary=[None] * count,
        gold_urgent=[row.is_urgent for row in validation],
        pred_urgent=[False] * count,
        urgent_proba=np.zeros(count, dtype=np.float64),
        languages=[row.language for row in validation],
        channels=[row.channel for row in validation],
    )
    metrics["majority_category"] = majority
    return metrics


def append_experiment(row: dict[str, Any], path: Path = EXPERIMENTS_PATH) -> None:
    """Append one experiment. Creates the file and header when missing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.is_file() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        payload = {column: row.get(column, "") for column in CSV_COLUMNS}
        if not payload["date"]:
            payload["date"] = date.today().isoformat()
        for column in CSV_COLUMNS:
            value = payload[column]
            if value is None:
                payload[column] = ""
            elif isinstance(value, float):
                payload[column] = f"{value:.6f}"
        writer.writerow(payload)


def record_majority(path: Path = EXPERIMENTS_PATH) -> dict[str, Any]:
    metrics = majority_metrics()
    append_experiment(
        {
            "id": "majority_class",
            "branch": "baseline",
            "features": "none",
            "hyperparameters": f"predict {metrics['majority_category']} for every ticket",
            "stage_a_accuracy": metrics["accuracy"],
            "stage_a_macro_f1": metrics["macro_f1"],
            "stage_a_urgent_f1": metrics["urgent_f1"],
            "stage_a_secondary_exact": metrics["secondary_exact"],
            "notes": "floor: train majority category, secondary null, urgent false, scored on validation",
        },
        path=path,
    )
    return metrics
