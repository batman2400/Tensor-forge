"""Classical baseline v0.

Three configurations, each scored two ways:

- Stage A: fit on train, score on validation (the honest held-out number).
- 5-fold OOF over train+validation, using the ids in ``ml/folds.json``.

The best configuration by OOF macro-F1 is refit on all 4,800 tickets and saved to
``artifacts/classical.joblib``. Thresholds stay at 0.5 for v0; consistency rules
still run, so the logged metrics are what the service would return.

    python -m ml.train_classical
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight

from app.inference import Engine, align_columns, apply_rules, positive_proba
from app.labels import CATEGORIES, SECONDARY_CATEGORIES
from app.text import build_input
from ml.data import TicketRow, load_all, load_folds
from ml.evaluate import REPORTS_DIR, append_experiment, compute_metrics

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "artifacts" / "classical.joblib"
REPRO_PATH = REPORTS_DIR / "repro_sample.json"
DECISION_PATH = REPORTS_DIR / "baseline_decision.md"

THRESHOLDS = {"secondary": 0.5, "urgent": 0.5, "review": 0.5}
MODEL_VERSION = "v0.1.0"


@dataclass(frozen=True)
class Config:
    id: str
    use_char: bool
    kind: str
    features: str
    hyperparameters: str


CONFIGS = (
    Config(
        id="logreg_word",
        use_char=False,
        kind="logreg",
        features="tfidf word 1-2 grams",
        hyperparameters="LogReg C=1 class_weight=balanced max_iter=1000; thresholds 0.5",
    ),
    Config(
        id="logreg_word_char",
        use_char=True,
        kind="logreg",
        features="tfidf word 1-2 + char_wb 2-5",
        hyperparameters="LogReg C=1 class_weight=balanced max_iter=1000; thresholds 0.5",
    ),
    Config(
        id="svc_word_char",
        use_char=True,
        kind="svc",
        features="tfidf word 1-2 + char_wb 2-5",
        hyperparameters="Calibrated LinearSVC C=1 class_weight=balanced ensemble=False; thresholds 0.5",
    ),
)


def _tfidf(
    analyzer: str, ngram_range: tuple[int, int], max_features: int, *, min_df: int = 2
) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        sublinear_tf=True,
        lowercase=False,
        dtype=np.float32,
    )


def make_vectorizer(
    use_char: bool,
    *,
    word_ngram: tuple[int, int] = (1, 2),
    char_ngram: tuple[int, int] = (2, 5),
    word_max: int = 40_000,
    char_max: int = 30_000,
    char_analyzer: str = "char_wb",
    min_df: int = 2,
) -> TfidfVectorizer | FeatureUnion:
    word = _tfidf("word", word_ngram, word_max, min_df=min_df)
    if not use_char:
        return word
    char = _tfidf(char_analyzer, char_ngram, char_max, min_df=min_df)
    return FeatureUnion([("word", word), ("char", char)])


def make_classifier(kind: str, y: np.ndarray, *, C: float = 1.0):
    """A fresh estimator. SVC falls back to logistic regression if a class is too rare to calibrate."""
    logreg = LogisticRegression(
        C=C,
        class_weight="balanced",
        max_iter=1000 if C == 1.0 else 2000,
        solver="lbfgs",
        random_state=42,
    )
    if kind == "logreg":
        return logreg
    if kind == "nb":
        return ComplementNB(alpha=0.5)
    if kind == "sgd":
        return SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            class_weight="balanced",
            max_iter=40,
            tol=1e-4,
            random_state=42,
        )
    if kind == "lgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=120,
            learning_rate=0.1,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.3,
            class_weight="balanced",
            random_state=42,
            n_jobs=4,
            verbosity=-1,
            force_col_wise=True,
        )
    _labels, counts = np.unique(y, return_counts=True)
    if len(counts) < 2 or int(counts.min()) < 2:
        return clone(logreg)
    splits = min(3, int(counts.min()))
    return CalibratedClassifierCV(
        LinearSVC(
            C=C,
            class_weight="balanced",
            dual="auto",
            max_iter=4000,
            random_state=42,
        ),
        method="sigmoid",
        cv=StratifiedKFold(n_splits=splits, shuffle=True, random_state=42),
        ensemble=False,
    )


def _fit_estimator(model, matrix, y: np.ndarray, kind: str) -> None:
    if kind == "nb":
        weights = compute_sample_weight(class_weight="balanced", y=y)
        model.fit(matrix, y, sample_weight=weights)
        return
    model.fit(matrix, y)


def fit_heads(matrix, row_index: list[int], rows: list[TicketRow], kind: str, *, C: float = 1.0):
    y_cat = np.array([rows[i].category for i in row_index])
    category_model = make_classifier(kind, y_cat, C=C)
    _fit_estimator(category_model, matrix, y_cat, kind)

    secondary_models = {}
    for label in SECONDARY_CATEGORIES:
        y_sec = np.array([1 if rows[i].secondary_category == label else 0 for i in row_index])
        if int(y_sec.min()) == int(y_sec.max()):
            secondary_models[label] = None
            continue
        model = make_classifier(kind, y_sec, C=C)
        _fit_estimator(model, matrix, y_sec, kind)
        secondary_models[label] = model

    y_urgent = np.array([1 if rows[i].is_urgent else 0 for i in row_index])
    urgent_model = make_classifier(kind, y_urgent, C=C)
    _fit_estimator(urgent_model, matrix, y_urgent, kind)
    return category_model, secondary_models, urgent_model


def head_probabilities(category_model, secondary_models, urgent_model, matrix):
    """Category, secondary, and urgency probabilities aligned to the label lists."""
    category_proba = align_columns(
        category_model.predict_proba(matrix),
        list(category_model.classes_),
        list(CATEGORIES),
    )
    width = matrix.shape[0]
    secondary_columns = []
    for label in SECONDARY_CATEGORIES:
        model = secondary_models[label]
        if model is None:
            secondary_columns.append(np.zeros(width, dtype=np.float64))
        else:
            secondary_columns.append(positive_proba(model, matrix))
    secondary_proba = np.column_stack(secondary_columns)
    urgent_proba = positive_proba(urgent_model, matrix)
    return category_proba, secondary_proba, urgent_proba


def predict_matrix(category_model, secondary_models, urgent_model, matrix, thresholds=None):
    category_proba, secondary_proba, urgent_proba = head_probabilities(
        category_model, secondary_models, urgent_model, matrix
    )
    used = THRESHOLDS if thresholds is None else thresholds
    labels = list(CATEGORIES)
    secondary_labels = list(SECONDARY_CATEGORIES)
    predictions = [
        apply_rules(
            category_proba[i],
            labels,
            secondary_proba[i],
            secondary_labels,
            float(urgent_proba[i]),
            used,
        )
        for i in range(category_proba.shape[0])
    ]
    return predictions, category_proba, urgent_proba


def _metrics(rows: list[TicketRow], indexes: list[int], predictions, category_proba, urgent_proba):
    return compute_metrics(
        gold_category=[rows[i].category for i in indexes],
        pred_category=[predictions[i]["category"] for i in range(len(indexes))],
        category_proba=category_proba,
        gold_secondary=[rows[i].secondary_category for i in indexes],
        pred_secondary=[predictions[i]["secondary_category"] for i in range(len(indexes))],
        gold_urgent=[rows[i].is_urgent for i in indexes],
        pred_urgent=[predictions[i]["is_urgent"] for i in range(len(indexes))],
        urgent_proba=urgent_proba,
        languages=[rows[i].language for i in indexes],
        channels=[rows[i].channel for i in indexes],
    )


def _fit_transform(texts: list[str], train_index: list[int], val_index: list[int], use_char: bool):
    vectorizer = make_vectorizer(use_char)
    x_train = vectorizer.fit_transform([texts[i] for i in train_index])
    x_val = vectorizer.transform([texts[i] for i in val_index])
    return vectorizer, x_train, x_val


def evaluate_config(
    config: Config,
    rows: list[TicketRow],
    texts: list[str],
    folds: dict[str, int],
    *,
    skip_cv: bool,
    stage_cache: dict,
    cv_cache: dict,
) -> dict:
    started = time.perf_counter()
    train_index = [i for i, row in enumerate(rows) if row.split == "train"]
    val_index = [i for i, row in enumerate(rows) if row.split == "validation"]

    if config.use_char not in stage_cache:
        print(f"  fitting stage-A vectorizer use_char={config.use_char}", flush=True)
        stage_cache[config.use_char] = _fit_transform(
            texts, train_index, val_index, config.use_char
        )
    _vectorizer, x_train, x_val = stage_cache[config.use_char]
    print(f"  stage A heads {config.id}", flush=True)
    category_model, secondary_models, urgent_model = fit_heads(
        x_train, train_index, rows, config.kind
    )
    predictions, category_proba, urgent_proba = predict_matrix(
        category_model, secondary_models, urgent_model, x_val
    )
    stage_a = _metrics(rows, val_index, predictions, category_proba, urgent_proba)
    print(
        f"  stage A {config.id}: acc {stage_a['accuracy']:.4f} macroF1 {stage_a['macro_f1']:.4f} "
        f"sec {stage_a['secondary_exact']:.4f} urgF1 {stage_a['urgent_f1']:.4f}",
        flush=True,
    )

    cv_metrics = None
    if not skip_cv:
        if config.use_char not in cv_cache:
            print(f"  fitting 5 fold vectorizers use_char={config.use_char}", flush=True)
            folded = []
            for fold in range(5):
                fold_train = [i for i, row in enumerate(rows) if folds[row.ticket_id] != fold]
                fold_val = [i for i, row in enumerate(rows) if folds[row.ticket_id] == fold]
                t0 = time.perf_counter()
                _vec, x_tr, x_va = _fit_transform(texts, fold_train, fold_val, config.use_char)
                folded.append((fold_train, fold_val, x_tr, x_va))
                print(f"    fold {fold} vectorizer in {time.perf_counter() - t0:.1f}s", flush=True)
            cv_cache[config.use_char] = folded

        oof_pred: list[dict | None] = [None] * len(rows)
        oof_cat = np.zeros((len(rows), len(CATEGORIES)), dtype=np.float64)
        oof_urg = np.zeros(len(rows), dtype=np.float64)
        for fold_train, fold_val, x_tr, x_va in cv_cache[config.use_char]:
            t0 = time.perf_counter()
            category_model, secondary_models, urgent_model = fit_heads(
                x_tr, fold_train, rows, config.kind
            )
            predictions, category_proba, urgent_proba = predict_matrix(
                category_model, secondary_models, urgent_model, x_va
            )
            for local, global_index in enumerate(fold_val):
                oof_pred[global_index] = predictions[local]
                oof_cat[global_index] = category_proba[local]
                oof_urg[global_index] = urgent_proba[local]
            print(f"    {config.id} fold heads in {time.perf_counter() - t0:.1f}s", flush=True)
        if any(item is None for item in oof_pred):
            raise RuntimeError("OOF predictions were not filled for every ticket")
        everything = list(range(len(rows)))
        cv_metrics = _metrics(rows, everything, oof_pred, oof_cat, oof_urg)
        print(
            f"  CV {config.id}: acc {cv_metrics['accuracy']:.4f} macroF1 {cv_metrics['macro_f1']:.4f} "
            f"sec {cv_metrics['secondary_exact']:.4f} urgF1 {cv_metrics['urgent_f1']:.4f}",
            flush=True,
        )

    elapsed = time.perf_counter() - started
    result = {
        "config": config,
        "stage_a": stage_a,
        "cv": cv_metrics,
        "seconds": round(elapsed, 1),
    }
    (REPORTS_DIR / f"{config.id}.json").write_text(
        json.dumps(
            {"id": config.id, "seconds": result["seconds"], "stage_a": stage_a, "cv": cv_metrics},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def _save_artifact(config: Config, rows: list[TicketRow], texts: list[str]) -> None:
    print(f"refitting {config.id} on all {len(rows)} tickets", flush=True)
    vectorizer = make_vectorizer(config.use_char)
    matrix = vectorizer.fit_transform(texts)
    everything = list(range(len(rows)))
    category_model, secondary_models, urgent_model = fit_heads(
        matrix, everything, rows, config.kind
    )
    missing = [label for label, model in secondary_models.items() if model is None]
    if missing:
        raise RuntimeError(f"secondary class missing from the full fit: {missing}")
    bundle = {
        "format": 1,
        "model_version": MODEL_VERSION,
        "config_name": config.id,
        "vectorizer": vectorizer,
        "category_model": category_model,
        "category_labels": list(CATEGORIES),
        "secondary_models": secondary_models,
        "secondary_labels": list(SECONDARY_CATEGORIES),
        "urgent_model": urgent_model,
        "thresholds": dict(THRESHOLDS),
        "trained_on": "train+validation",
        "n_rows": len(rows),
    }
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, ARTIFACT_PATH, compress=3)
    print(f"wrote {ARTIFACT_PATH}", flush=True)

    engine = Engine.load(ARTIFACT_PATH)
    sample_rows = sorted(
        (row for row in rows if row.split == "validation"), key=lambda row: row.ticket_id
    )[:12]
    sample = []
    for row in sample_rows:
        prediction = engine.predict(
            {
                "channel": row.channel,
                "subject": row.subject,
                "text": row.text,
                "ticket_id": row.ticket_id,
            }
        )
        sample.append(
            {
                "ticket_id": row.ticket_id,
                "category": prediction["category"],
                "secondary_category": prediction["secondary_category"],
                "is_urgent": prediction["is_urgent"],
                "confidence": prediction["confidence"],
                "team": prediction["team"],
                "model_version": prediction["model_version"],
            }
        )
    REPRO_PATH.write_text(json.dumps(sample, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {REPRO_PATH}", flush=True)


def _write_decision(results: list[dict], best: Config) -> None:
    ranked = sorted(results, key=lambda item: -item["stage_a"]["macro_f1"])
    lines = [
        "# Classical baseline v0 decision",
        "",
        "Stage A (fit on train, score on validation) is the number to quote.",
        "Random 5-fold scores on the pooled 4,800 tickets are much higher because tickets share",
        "opening templates across folds, and the organizer split has no exact text overlap.",
        "OOF is only a ranking check. Thresholds are 0.5 (not tuned). Review threshold is 0.5.",
        "",
        "| config | stage A acc | stage A macro-F1 | stage A urgent F1 | stage A secondary exact | CV macro-F1 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in results:
        stage_a = item["stage_a"]
        cv_f1 = "" if item["cv"] is None else f"{item['cv']['macro_f1']:.4f}"
        lines.append(
            f"| {item['config'].id} | {stage_a['accuracy']:.4f} | {stage_a['macro_f1']:.4f} | "
            f"{stage_a['urgent_f1']:.4f} | {stage_a['secondary_exact']:.4f} | {cv_f1} |"
        )
    winner = next(item for item in results if item["config"].id == best.id)
    others = [item for item in ranked if item["config"].id != best.id]
    runner = others[0] if others else None
    if runner is not None:
        margin = winner["stage_a"]["macro_f1"] - runner["stage_a"]["macro_f1"]
        reason = (
            f"Chose `{best.id}` on Stage A macro-F1 {winner['stage_a']['macro_f1']:.4f} "
            f"(accuracy {winner['stage_a']['accuracy']:.4f}), {margin:.4f} above `{runner['config'].id}`. "
        )
    else:
        reason = (
            f"Chose `{best.id}` on Stage A macro-F1 {winner['stage_a']['macro_f1']:.4f} "
            f"(accuracy {winner['stage_a']['accuracy']:.4f}). "
        )
    if winner["cv"] is not None and runner is not None and runner["cv"] is not None:
        if winner["cv"]["macro_f1"] >= runner["cv"]["macro_f1"]:
            reason += (
                f"OOF macro-F1 ranks it the same way ({winner['cv']['macro_f1']:.4f} vs "
                f"{runner['cv']['macro_f1']:.4f}), but those OOF scores are not the accuracy to publish. "
            )
        else:
            reason += (
                f"OOF macro-F1 prefers `{runner['config'].id}` ({runner['cv']['macro_f1']:.4f} vs "
                f"{winner['cv']['macro_f1']:.4f}); Stage A still decides, because the organizer split is harder. "
            )
    if (
        runner is not None
        and winner["stage_a"]["urgent_f1"] + 1e-9 < runner["stage_a"]["urgent_f1"]
    ):
        reason += (
            f"Stage A urgent F1 is {winner['stage_a']['urgent_f1']:.4f}, below "
            f"`{runner['config'].id}` at {runner['stage_a']['urgent_f1']:.4f}. "
            "Category macro-F1 is the selection key; the urgency threshold stays at 0.5 for day 2. "
        )
    reason += (
        "The serving artifact is this configuration refit on all 4,800 tickets, version v0.1.0. "
        "Thresholds are deliberately untuned at 0.5 so day-2 error analysis can move them "
        "without mixing that choice into v0."
    )
    lines.extend(["", reason, ""])
    DECISION_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(reason, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-cv", action="store_true", help="stage A only (debug)")
    parser.add_argument("--configs", nargs="*", choices=[config.id for config in CONFIGS])
    args = parser.parse_args()

    selected = set(args.configs or [config.id for config in CONFIGS])
    configs = [config for config in CONFIGS if config.id in selected]
    rows = load_all()
    folds = load_folds()
    missing = [row.ticket_id for row in rows if row.ticket_id not in folds]
    if missing:
        raise SystemExit(
            f"{len(missing)} tickets missing from ml/folds.json; run python -m ml.data"
        )

    texts = [build_input(row.channel, row.subject, row.text) for row in rows]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stage_cache: dict = {}
    cv_cache: dict = {}
    results = []
    for config in configs:
        print(f"== {config.id}", flush=True)
        results.append(
            evaluate_config(
                config,
                rows,
                texts,
                folds,
                skip_cv=args.skip_cv,
                stage_cache=stage_cache,
                cv_cache=cv_cache,
            )
        )

    def sort_key(item: dict) -> tuple:
        cv_f1 = -1.0 if item["cv"] is None else item["cv"]["macro_f1"]
        # Stage A is the honest split. OOF is a tie-break only: random folds share templates.
        simplicity = {"logreg_word": 2, "logreg_word_char": 1, "svc_word_char": 0}[
            item["config"].id
        ]
        return (item["stage_a"]["macro_f1"], cv_f1, simplicity)

    best = max(results, key=sort_key)["config"]
    for item in results:
        cv = item["cv"]
        chosen = item["config"].id == best.id
        append_experiment(
            {
                "id": item["config"].id,
                "branch": "classical",
                "features": item["config"].features,
                "hyperparameters": item["config"].hyperparameters,
                "stage_a_accuracy": item["stage_a"]["accuracy"],
                "stage_a_macro_f1": item["stage_a"]["macro_f1"],
                "stage_a_urgent_f1": item["stage_a"]["urgent_f1"],
                "stage_a_secondary_exact": item["stage_a"]["secondary_exact"],
                "cv_accuracy": None if cv is None else cv["accuracy"],
                "cv_macro_f1": None if cv is None else cv["macro_f1"],
                "cv_urgent_f1": None if cv is None else cv["urgent_f1"],
                "cv_secondary_exact": None if cv is None else cv["secondary_exact"],
                "notes": "day-1 v0; rules applied; thresholds 0.5"
                + ("; selected for the serving artifact" if chosen else ""),
            }
        )

    _write_decision(results, best)
    stage_cache.clear()
    cv_cache.clear()
    gc.collect()
    _save_artifact(best, rows, texts)


if __name__ == "__main__":
    main()
