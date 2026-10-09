"""Save svc_word_char probabilities for fusion.

Stage A is fit on train and scored on validation. The out-of-fold file has one
row per ticket in ``load_all()`` order, from the fold that held that ticket out.
This does not rewrite ``artifacts/classical.joblib`` or ``experiments.csv``.

    python -m ml.export_classical_oof
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from app.labels import CATEGORIES, SECONDARY_CATEGORIES
from app.text import build_input
from ml.data import load_all, load_folds
from ml.thresholds import decide
from ml.train_classical import (
    CONFIGS,
    THRESHOLDS,
    _fit_transform,
    _metrics,
    fit_heads,
    head_probabilities,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "ml" / "runs" / "classical_svc_word_char"
EXPECTED_STAGE_A_MACRO_F1 = 0.655821


def _save(path: Path, ticket_ids: list[str], category, secondary, urgent) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ticket_id=np.array(ticket_ids),
        category_proba=np.asarray(category, dtype=np.float64),
        secondary_proba=np.asarray(secondary, dtype=np.float64),
        urgent_proba=np.asarray(urgent, dtype=np.float64),
    )
    print(f"wrote {path} n={len(ticket_ids)}", flush=True)


def main() -> None:
    config = next(item for item in CONFIGS if item.id == "svc_word_char")
    rows = load_all()
    texts = [build_input(row.channel, row.subject, row.text) for row in rows]
    folds = load_folds(ROOT / "ml" / "folds.json")
    train_index = [i for i, row in enumerate(rows) if row.split == "train"]
    val_index = [i for i, row in enumerate(rows) if row.split == "validation"]

    print("stage A vectorizer", flush=True)
    _vectorizer, x_train, x_val = _fit_transform(texts, train_index, val_index, config.use_char)
    print("stage A heads", flush=True)
    category_model, secondary_models, urgent_model = fit_heads(
        x_train, train_index, rows, config.kind
    )
    category_proba, secondary_proba, urgent_proba = head_probabilities(
        category_model, secondary_models, urgent_model, x_val
    )
    predictions = decide(category_proba, secondary_proba, urgent_proba, THRESHOLDS)
    stage_a = _metrics(rows, val_index, predictions, category_proba, urgent_proba)
    print(
        f"stage A macroF1 {stage_a['macro_f1']:.6f} (logged {EXPECTED_STAGE_A_MACRO_F1:.6f})",
        flush=True,
    )
    if abs(stage_a["macro_f1"] - EXPECTED_STAGE_A_MACRO_F1) > 0.002:
        raise SystemExit("stage A macro-F1 does not match the logged svc_word_char run")
    _save(
        OUT_DIR / "stage_a" / "proba.npz",
        [rows[i].ticket_id for i in val_index],
        category_proba,
        secondary_proba,
        urgent_proba,
    )
    del category_model, secondary_models, urgent_model, x_train, x_val

    n = len(rows)
    oof_cat = np.zeros((n, len(CATEGORIES)), dtype=np.float64)
    oof_sec = np.zeros((n, len(SECONDARY_CATEGORIES)), dtype=np.float64)
    oof_urg = np.zeros(n, dtype=np.float64)
    filled = np.zeros(n, dtype=bool)
    for fold in range(5):
        fold_train = [i for i, row in enumerate(rows) if folds[row.ticket_id] != fold]
        fold_val = [i for i, row in enumerate(rows) if folds[row.ticket_id] == fold]
        started = time.perf_counter()
        print(f"fold {fold} vectorizer n_val={len(fold_val)}", flush=True)
        _vectorizer, x_tr, x_va = _fit_transform(texts, fold_train, fold_val, config.use_char)
        print(f"fold {fold} heads", flush=True)
        category_model, secondary_models, urgent_model = fit_heads(
            x_tr, fold_train, rows, config.kind
        )
        category_proba, secondary_proba, urgent_proba = head_probabilities(
            category_model, secondary_models, urgent_model, x_va
        )
        for local, global_index in enumerate(fold_val):
            oof_cat[global_index] = category_proba[local]
            oof_sec[global_index] = secondary_proba[local]
            oof_urg[global_index] = urgent_proba[local]
            filled[global_index] = True
        print(f"fold {fold} done in {time.perf_counter() - started:.1f}s", flush=True)
        del category_model, secondary_models, urgent_model, x_tr, x_va
    if not bool(filled.all()):
        raise SystemExit(f"{int((~filled).sum())} tickets have no classical OOF row")
    _save(
        OUT_DIR / "oof" / "proba.npz",
        [row.ticket_id for row in rows],
        oof_cat,
        oof_sec,
        oof_urg,
    )


if __name__ == "__main__":
    main()
