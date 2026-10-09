"""Reliability diagram for the frozen fusion settings on Stage A.

Stage A trains on the train split and scores validation. That is the honest
figure. The shipped encoder was later fit on all 4,800 tickets, so scoring
that file on validation would look better than it is.

    python -m ml.calibration
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.fusion import apply_fusion
from ml.data import load_all
from ml.error_analysis import rule_masks, slice_summary
from ml.evaluate import REPORTS_DIR, compute_metrics, reliability_bins
from ml.fuse import CLASSICAL_DIR, ENCODER_DIR, _align, _bundle
from ml.thresholds import decide

DIAGRAM_PATH = REPORTS_DIR / "calibration.svg"
REPORT_PATH = REPORTS_DIR / "calibration.md"
SETTINGS_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "fusion.json"


def _settings() -> dict:
    payload = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    settings = payload.get("settings", payload)
    if "encoder_weight" not in settings:
        raise RuntimeError("fusion settings are missing encoder_weight")
    return settings


def stage_a_fusion() -> tuple[dict, list[dict], list]:
    """Frozen fusion settings applied to the Stage A probability files."""
    rows = [row for row in load_all() if row.split == "validation"]
    ticket_ids = [row.ticket_id for row in rows]
    classical = _align(_bundle(CLASSICAL_DIR / "stage_a" / "proba.npz"), ticket_ids)
    encoder = _align(_bundle(ENCODER_DIR / "stage_a" / "proba.npz"), ticket_ids)
    settings = _settings()
    category, secondary, urgent = apply_fusion(classical, encoder, settings)
    predictions = decide(category, secondary, urgent, settings["thresholds"])
    metrics = compute_metrics(
        gold_category=[row.category for row in rows],
        pred_category=[item["category"] for item in predictions],
        category_proba=category,
        gold_secondary=[row.secondary_category for row in rows],
        pred_secondary=[item["secondary_category"] for item in predictions],
        gold_urgent=[row.is_urgent for row in rows],
        pred_urgent=[bool(item["is_urgent"]) for item in predictions],
        urgent_proba=urgent,
        languages=[row.language for row in rows],
        channels=[row.channel for row in rows],
    )
    confidence = category.max(axis=1)
    correct = np.array(
        [item["category"] == row.category for item, row in zip(predictions, rows, strict=True)],
        dtype=np.float64,
    )
    return (
        metrics,
        reliability_bins(confidence, correct, n_bins=10),
        [(name, slice_summary(rows, predictions, mask)) for name, mask in rule_masks(rows).items()],
    )


def _svg(bins: list[dict]) -> str:
    width, height = 440, 300
    left, top, plot = 48, 16, 240
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="48" y="14" font-family="sans-serif" font-size="13">'
        "Stage A reliability: confidence vs accuracy</text>",
    ]
    x0, y0 = left, top + 12
    parts.append(
        f'<line x1="{x0}" y1="{y0 + plot}" x2="{x0 + plot}" y2="{y0}" '
        'stroke="#bbbbbb" stroke-dasharray="4 3"/>'
    )
    slot = plot / max(len(bins), 1)
    for index, item in enumerate(bins):
        if not item["n"] or item["accuracy"] is None:
            continue
        bar_h = item["accuracy"] * plot
        x = x0 + index * slot + 4
        y = y0 + plot - bar_h
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{slot - 8:.1f}" height="{bar_h:.1f}" '
            'fill="#2f6f4e"/>'
        )
    parts.append(
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + plot}" stroke="#222"/>'
        f'<line x1="{x0}" y1="{y0 + plot}" x2="{x0 + plot}" y2="{y0 + plot}" stroke="#222"/>'
    )
    parts.append(
        f'<text x="{x0}" y="{height - 8}" font-family="sans-serif" font-size="11">'
        "confidence bin →</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts)


def _markdown(metrics: dict, bins: list[dict], slices: list[tuple]) -> str:
    lines = [
        "# Calibration (Stage A fusion)",
        "",
        "Weights and temperatures are the frozen settings in `artifacts/fusion.json`.",
        "They were fit on out-of-fold probabilities. This table scores the train-only",
        "models on the 800 validation tickets. The shipped encoder was fit again on",
        "all 4,800 tickets, so a score of that file on validation is not this number.",
        "",
        f"Accuracy {metrics['accuracy']:.4f}, macro-F1 {metrics['macro_f1']:.4f}, "
        f"ECE {metrics['ece']:.4f}, Brier {metrics['brier']:.4f}.",
        "",
        "| bin | n | mean confidence | accuracy |",
        "| --- | --- | --- | --- |",
    ]
    for item in bins:
        if not item["n"]:
            lines.append(f"| {item['low']:.1f}–{item['high']:.1f} | 0 |  |  |")
            continue
        lines.append(
            f"| {item['low']:.1f}–{item['high']:.1f} | {item['n']} | "
            f"{item['confidence']:.3f} | {item['accuracy']:.3f} |"
        )
    lines.extend(
        [
            "",
            "The diagram is `calibration.svg`. A bin on the diagonal is calibrated.",
            "Leave-one-language-out is not in this table: that would retrain without",
            "each language. These are slices of the same Stage A predictions.",
            "",
            "| language | n | accuracy | macro-F1 |",
            "| --- | --- | --- | --- |",
        ]
    )
    for language, row in metrics["by_language"].items():
        lines.append(f"| {language} | {row['n']} | {row['accuracy']:.3f} | {row['macro_f1']:.3f} |")
    lines.extend(
        [
            "",
            "| channel | n | accuracy | macro-F1 |",
            "| --- | --- | --- | --- |",
            *[
                f"| {channel} | {row['n']} | {row['accuracy']:.3f} | {row['macro_f1']:.3f} |"
                for channel, row in metrics["by_channel"].items()
            ],
            "",
            "| slice | n | accuracy | secondary exact | urgent FP rate |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for name, summary in slices:
        lines.append(
            f"| {name} | {summary['n']} | {summary['accuracy']:.3f} | "
            f"{summary['secondary_exact']:.3f} | {summary['urgent_false_positive_rate']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    metrics, bins, slices = stage_a_fusion()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DIAGRAM_PATH.write_text(_svg(bins), encoding="utf-8")
    REPORT_PATH.write_text(_markdown(metrics, bins, slices), encoding="utf-8")
    print(
        f"stage A macro-F1 {metrics['macro_f1']:.4f} ece {metrics['ece']:.4f}",
        flush=True,
    )
    print(f"wrote {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
