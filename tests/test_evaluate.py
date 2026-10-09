from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.labels import CATEGORIES
from ml.calibration import stage_a_fusion
from ml.evaluate import METRIC_KEYS, compute_metrics, majority_metrics, reliability_bins


def test_reliability_bins_cover_every_row_and_the_top_bin():
    confidence = np.array([0.05, 0.95, 1.0])
    correct = np.array([0.0, 1.0, 1.0])
    bins = reliability_bins(confidence, correct, n_bins=2)
    assert sum(item["n"] for item in bins) == 3
    assert bins[0]["n"] == 1
    assert bins[0]["accuracy"] == 0.0
    assert bins[1]["n"] == 2
    assert bins[1]["accuracy"] == 1.0


def test_compute_metrics_on_a_perfect_majority_slice():
    labels = list(CATEGORIES)
    count = 4
    proba = np.zeros((count, len(labels)))
    proba[:, 0] = 1.0
    gold = ["payment_refund"] * count
    metrics = compute_metrics(
        gold_category=gold,
        pred_category=gold,
        category_proba=proba,
        gold_secondary=[None] * count,
        pred_secondary=[None] * count,
        gold_urgent=[False] * count,
        pred_urgent=[False] * count,
        urgent_proba=np.zeros(count),
        languages=["en", "en", "si", "si"],
        channels=["chat", "email", "chat", "email"],
    )
    assert set(METRIC_KEYS) <= set(metrics)
    assert metrics["n"] == count
    assert metrics["accuracy"] == 1.0
    assert metrics["ece"] == 0.0
    assert metrics["brier"] == 0.0
    assert metrics["secondary_exact"] == 1.0
    assert abs(metrics["macro_f1"] - (1 / len(labels))) < 1e-5
    assert len(metrics["confusion_matrix"]) == len(labels)
    assert metrics["by_language"]["en"]["n"] == 2
    assert metrics["by_channel"]["chat"]["accuracy"] == 1.0


@pytest.mark.skipif(
    not Path("ml/runs/encoder_e5_small/stage_a/proba.npz").is_file()
    or not Path("artifacts/fusion.json").is_file(),
    reason="stage A probabilities are not in the tree",
)
def test_frozen_fusion_stage_a_stays_above_the_gate():
    metrics, bins, _slices = stage_a_fusion()
    assert metrics["macro_f1"] >= 0.79
    assert metrics["ece"] < 0.15
    assert sum(item["n"] for item in bins) == metrics["n"] == 800


@pytest.mark.skipif(not Path("data/train.csv").is_file(), reason="dataset not downloaded")
def test_majority_baseline_fills_the_table():
    metrics = majority_metrics()
    assert set(METRIC_KEYS) <= set(metrics)
    assert metrics["n"] == 800
    assert metrics["majority_category"] == "payment_refund"
    assert 0 < metrics["accuracy"] < 0.5
    assert metrics["macro_f1"] < metrics["accuracy"]
    assert metrics["ece"] > 0.5
    assert set(metrics["by_language"]) == {"en", "si", "ta", "singlish", "tanglish", "mixed"}
    assert set(metrics["by_channel"]) == {"email", "chat", "call_transcript"}
