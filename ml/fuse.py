"""Fit fusion weights and temperatures on out-of-fold probabilities.

The mix is ``(1 - w) * classical + w * encoder`` after each branch is temperature
scaled. Weights and temperatures are chosen on OOF rows only. Thresholds are then
chosen on those fused OOF probabilities. Stage A (train fit, validation score) is
scored once with the frozen settings. Because OOF tuning includes validation
labels, Stage A is development validation, not an untouched holdout estimate.

This does not replace ``artifacts/classical.joblib``.

    python -m ml.fuse
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from app.fusion import apply_fusion, mix, scale_binary, scale_multiclass
from app.labels import CATEGORIES, SECONDARY_CATEGORIES
from ml.data import load_all
from ml.evaluate import append_experiment, compute_metrics, expected_calibration_error
from ml.thresholds import select_thresholds
from ml.train_classical import THRESHOLDS, _metrics

ROOT = Path(__file__).resolve().parents[1]
CLASSICAL_DIR = ROOT / "ml" / "runs" / "classical_svc_word_char"
ENCODER_DIR = ROOT / "ml" / "runs" / "encoder_e5_small"
REPORT_JSON = ROOT / "ml" / "reports" / "fusion.json"
REPORT_MD = ROOT / "ml" / "reports" / "fusion.md"

WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
TEMPERATURES = (0.5, 1.0, 1.5, 2.0)


def _bundle(path: Path) -> dict[str, np.ndarray]:
    payload = np.load(path, allow_pickle=False)
    return {
        "ticket_id": np.array([str(item) for item in payload["ticket_id"].tolist()]),
        "category_proba": np.asarray(payload["category_proba"], dtype=np.float64),
        "secondary_proba": np.asarray(payload["secondary_proba"], dtype=np.float64),
        "urgent_proba": np.asarray(payload["urgent_proba"], dtype=np.float64),
    }


def _align(source: dict[str, np.ndarray], ticket_ids: list[str]) -> dict[str, np.ndarray]:
    position = {ticket_id: index for index, ticket_id in enumerate(source["ticket_id"].tolist())}
    missing = [ticket_id for ticket_id in ticket_ids if ticket_id not in position]
    if missing:
        raise RuntimeError(f"{len(missing)} tickets missing from a probability file")
    index = np.array([position[ticket_id] for ticket_id in ticket_ids])
    return {
        "ticket_id": np.array(ticket_ids),
        "category_proba": source["category_proba"][index],
        "secondary_proba": source["secondary_proba"][index],
        "urgent_proba": source["urgent_proba"][index],
    }


def _encoder_oof(ticket_ids: list[str]) -> dict[str, np.ndarray]:
    n = len(ticket_ids)
    category = np.zeros((n, len(CATEGORIES)), dtype=np.float64)
    secondary = np.zeros((n, len(SECONDARY_CATEGORIES)), dtype=np.float64)
    urgent = np.zeros(n, dtype=np.float64)
    filled = np.zeros(n, dtype=bool)
    position = {ticket_id: index for index, ticket_id in enumerate(ticket_ids)}
    for fold in range(5):
        payload = _bundle(ENCODER_DIR / f"fold_{fold}" / "proba.npz")
        for row_index, ticket_id in enumerate(payload["ticket_id"].tolist()):
            slot = position[ticket_id]
            category[slot] = payload["category_proba"][row_index]
            secondary[slot] = payload["secondary_proba"][row_index]
            urgent[slot] = payload["urgent_proba"][row_index]
            filled[slot] = True
    if not bool(filled.all()):
        raise RuntimeError(f"{int((~filled).sum())} tickets have no encoder OOF row")
    return {
        "ticket_id": np.array(ticket_ids),
        "category_proba": category,
        "secondary_proba": secondary,
        "urgent_proba": urgent,
    }


def _category_f1(proba: np.ndarray, gold: list[str]) -> float:
    predicted = [CATEGORIES[int(index)] for index in proba.argmax(axis=1)]
    return float(
        f1_score(gold, predicted, average="macro", labels=list(CATEGORIES), zero_division=0)
    )


def _ece(proba: np.ndarray, gold: list[str]) -> float:
    predicted = [CATEGORIES[int(index)] for index in proba.argmax(axis=1)]
    correct = np.array([pred == label for pred, label in zip(predicted, gold, strict=True)])
    return float(expected_calibration_error(proba.max(axis=1), correct.astype(np.float64)))


def _choose_weight(score_at) -> float:
    """Highest score. Ties keep the smaller encoder weight."""
    best_weight = 0.0
    best_score = -1.0
    for weight in WEIGHTS:
        score = score_at(weight)
        if score > best_score + 1e-12:
            best_score = score
            best_weight = weight
    return best_weight


def _fit_category(classical, encoder, gold: list[str]) -> tuple[float, float]:
    weight = _choose_weight(lambda item: _category_f1(mix(classical, encoder, item), gold))
    mixed = mix(classical, encoder, weight)
    best_temp = 1.0
    best_ece = _ece(scale_multiclass(mixed, 1.0), gold)
    for temperature in TEMPERATURES:
        ece = _ece(scale_multiclass(mixed, temperature), gold)
        if ece < best_ece - 1e-12 or (
            abs(ece - best_ece) <= 1e-12 and abs(temperature - 1.0) < abs(best_temp - 1.0)
        ):
            best_ece = ece
            best_temp = temperature
    return weight, best_temp


def _scaled_pair(classical, encoder, weight: float, temperature: float, kind: str):
    if kind == "multi":
        return mix(
            scale_multiclass(classical, temperature), scale_multiclass(encoder, temperature), weight
        )
    return mix(scale_binary(classical, temperature), scale_binary(encoder, temperature), weight)


def _fit_binary_weight(classical, encoder, score_at) -> tuple[float, float]:
    best = (0.0, 1.0)
    best_score = -1.0
    for weight in WEIGHTS:
        for temperature in TEMPERATURES:
            score = score_at(weight, temperature)
            closer = abs(temperature - 1.0) < abs(best[1] - 1.0)
            better = score > best_score + 1e-12
            tie = abs(score - best_score) <= 1e-12 and (
                weight < best[0] or (weight == best[0] and closer)
            )
            if better or tie:
                best_score = score
                best = (weight, temperature)
    return best


def _secondary_exact(category, secondary, urgent, gold_secondary, gold_urgent) -> float:
    from ml.thresholds import decide

    predictions = decide(category, secondary, urgent, THRESHOLDS)
    matches = sum(
        gold == pred["secondary_category"]
        for gold, pred in zip(gold_secondary, predictions, strict=True)
    )
    return matches / len(gold_secondary)


def _urgent_f1(category, secondary, urgent, gold_secondary, gold_urgent) -> float:
    from ml.evaluate import _binary_f1
    from ml.thresholds import decide

    predictions = decide(category, secondary, urgent, THRESHOLDS)
    _precision, _recall, f1 = _binary_f1(
        gold_urgent, [bool(item["is_urgent"]) for item in predictions]
    )
    return f1


def fit_fusion(classical: dict[str, np.ndarray], encoder: dict[str, np.ndarray], rows) -> dict:
    gold_category = [row.category for row in rows]
    gold_secondary = [row.secondary_category for row in rows]
    gold_urgent = [row.is_urgent for row in rows]
    category_weight, category_temperature = _fit_category(
        classical["category_proba"], encoder["category_proba"], gold_category
    )
    category = scale_multiclass(
        mix(classical["category_proba"], encoder["category_proba"], category_weight),
        category_temperature,
    )

    def secondary_score(weight: float, temperature: float) -> float:
        secondary = _scaled_pair(
            classical["secondary_proba"],
            encoder["secondary_proba"],
            weight,
            temperature,
            "binary",
        )
        return _secondary_exact(
            category, secondary, classical["urgent_proba"], gold_secondary, gold_urgent
        )

    secondary_weight, secondary_temperature = _fit_binary_weight(
        classical["secondary_proba"], encoder["secondary_proba"], secondary_score
    )
    secondary = _scaled_pair(
        classical["secondary_proba"],
        encoder["secondary_proba"],
        secondary_weight,
        secondary_temperature,
        "binary",
    )

    def urgent_score(weight: float, temperature: float) -> float:
        urgent = _scaled_pair(
            classical["urgent_proba"], encoder["urgent_proba"], weight, temperature, "binary"
        )
        return _urgent_f1(category, secondary, urgent, gold_secondary, gold_urgent)

    urgent_weight, urgent_temperature = _fit_binary_weight(
        classical["urgent_proba"], encoder["urgent_proba"], urgent_score
    )
    urgent = _scaled_pair(
        classical["urgent_proba"],
        encoder["urgent_proba"],
        urgent_weight,
        urgent_temperature,
        "binary",
    )
    thresholds, point = select_thresholds(category, secondary, urgent, gold_secondary, gold_urgent)
    return {
        "encoder_weight": {
            "category": category_weight,
            "secondary": secondary_weight,
            "urgent": urgent_weight,
        },
        "temperature": {
            "category": category_temperature,
            "secondary": secondary_temperature,
            "urgent": urgent_temperature,
        },
        "thresholds": thresholds,
        "oof_holdout": {
            "secondary_exact": point.secondary_exact,
            "urgent_f1": point.urgent_f1,
        },
    }


def _write_report(settings: dict, stage_a: dict, classical_stage_a: dict) -> None:
    weights = settings["encoder_weight"]
    temperatures = settings["temperature"]
    thresholds = settings["thresholds"]
    delta = stage_a["macro_f1"] - classical_stage_a["macro_f1"]
    lines = [
        "# Fusion",
        "",
        "Weights are the encoder's share of a weighted average with `svc_word_char`.",
        "They are fit on out-of-fold probabilities and labels across all 4,800 records, including validation. Stage A below is tuning-exposed validation, not an untouched holdout estimate.",
        "The serving manifest fuses this with the final encoder (`artifacts/manifest.json`).",
        "",
        f"- Category encoder weight {weights['category']}, temperature {temperatures['category']}.",
        f"- Secondary encoder weight {weights['secondary']}, temperature {temperatures['secondary']}.",
        f"- Urgent encoder weight {weights['urgent']}, temperature {temperatures['urgent']}.",
        (
            f"- Thresholds secondary {thresholds['secondary']}, urgent {thresholds['urgent']}, "
            f"review {thresholds['review']}."
        ),
        "",
        "| setup | stage A acc | stage A macro-F1 | stage A urgent F1 | stage A secondary exact |",
        "| --- | --- | --- | --- | --- |",
        (
            f"| svc_word_char | {classical_stage_a['accuracy']:.4f} | {classical_stage_a['macro_f1']:.4f} | "
            f"{classical_stage_a['urgent_f1']:.4f} | {classical_stage_a['secondary_exact']:.4f} |"
        ),
        (
            f"| fusion | {stage_a['accuracy']:.4f} | {stage_a['macro_f1']:.4f} | "
            f"{stage_a['urgent_f1']:.4f} | {stage_a['secondary_exact']:.4f} |"
        ),
        "",
        (
            f"Stage A macro-F1 changes by {delta:+.4f} versus `svc_word_char`. "
            "Laptop ONNX latency is in `onnx_latency.md`. "
            "Gate G1 passed; see `g1.md`."
        ),
        "",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    classical_oof_path = CLASSICAL_DIR / "oof" / "proba.npz"
    classical_stage_path = CLASSICAL_DIR / "stage_a" / "proba.npz"
    if not classical_oof_path.is_file() or not classical_stage_path.is_file():
        raise SystemExit("missing classical probabilities; run python -m ml.export_classical_oof")
    rows = load_all()
    ticket_ids = [row.ticket_id for row in rows]
    classical_oof = _align(_bundle(classical_oof_path), ticket_ids)
    encoder_oof = _encoder_oof(ticket_ids)
    settings = fit_fusion(classical_oof, encoder_oof, rows)

    validation = [row for row in rows if row.split == "validation"]
    val_ids = [row.ticket_id for row in validation]
    classical_stage = _align(_bundle(classical_stage_path), val_ids)
    encoder_stage = _align(_bundle(ENCODER_DIR / "stage_a" / "proba.npz"), val_ids)
    category, secondary, urgent = apply_fusion(classical_stage, encoder_stage, settings)
    from ml.thresholds import decide

    predictions = decide(category, secondary, urgent, settings["thresholds"])
    stage_a = _metrics(
        rows,
        [index for index, row in enumerate(rows) if row.split == "validation"],
        predictions,
        category,
        urgent,
    )
    classical_only = decide(
        classical_stage["category_proba"],
        classical_stage["secondary_proba"],
        classical_stage["urgent_proba"],
        THRESHOLDS,
    )
    classical_metrics = compute_metrics(
        gold_category=[row.category for row in validation],
        pred_category=[item["category"] for item in classical_only],
        category_proba=classical_stage["category_proba"],
        gold_secondary=[row.secondary_category for row in validation],
        pred_secondary=[item["secondary_category"] for item in classical_only],
        gold_urgent=[row.is_urgent for row in validation],
        pred_urgent=[bool(item["is_urgent"]) for item in classical_only],
        urgent_proba=classical_stage["urgent_proba"],
        languages=[row.language for row in validation],
        channels=[row.channel for row in validation],
    )
    payload = {"settings": settings, "stage_a": stage_a, "classical_stage_a": classical_metrics}
    REPORT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_report(settings, stage_a, classical_metrics)
    append_experiment(
        {
            "id": "fusion_svc_e5",
            "branch": "fusion",
            "features": "svc_word_char + multilingual-e5-small",
            "hyperparameters": (
                f"encoder weights cat={settings['encoder_weight']['category']} "
                f"sec={settings['encoder_weight']['secondary']} "
                f"urg={settings['encoder_weight']['urgent']}; "
                f"T cat={settings['temperature']['category']} "
                f"sec={settings['temperature']['secondary']} "
                f"urg={settings['temperature']['urgent']}; "
                f"thresholds secondary={settings['thresholds']['secondary']} "
                f"urgent={settings['thresholds']['urgent']}"
            ),
            "stage_a_accuracy": stage_a["accuracy"],
            "stage_a_macro_f1": stage_a["macro_f1"],
            "stage_a_urgent_f1": stage_a["urgent_f1"],
            "stage_a_secondary_exact": stage_a["secondary_exact"],
            "notes": "weights and temperatures fit on OOF only; stage A scored once; serving manifest fuses the final encoder",
        }
    )
    print(
        f"fusion stage A macroF1 {stage_a['macro_f1']:.4f} "
        f"sec {stage_a['secondary_exact']:.4f} urgF1 {stage_a['urgent_f1']:.4f}",
        flush=True,
    )
    print(f"wrote {REPORT_MD}", flush=True)


if __name__ == "__main__":
    main()
