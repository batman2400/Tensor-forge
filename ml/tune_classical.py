"""Stage A search for a classical v1 candidate.

Thresholds are chosen on a stratified 20% holdout of train, then frozen and
scored once on validation. The day-1 serving file is left untouched.

    python -m ml.tune_classical
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from app.labels import CATEGORIES, SECONDARY_CATEGORIES
from app.text import build_input
from ml.data import TicketRow, load_all
from ml.error_analysis import write_report
from ml.evaluate import REPORTS_DIR, append_experiment
from ml.thresholds import decide, select_thresholds
from ml.train_classical import (
    THRESHOLDS,
    _metrics,
    fit_heads,
    head_probabilities,
    make_vectorizer,
)

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_PATH = ROOT / "artifacts" / "classical_v1.joblib"
DECISION_PATH = REPORTS_DIR / "classical_v1.md"
V0_MACRO_F1 = 0.655821
CANDIDATE_VERSION = "v0.2.0"


@dataclass(frozen=True)
class Spec:
    id: str
    kind: str
    features: str
    hyperparameters: str
    use_char: bool = True
    C: float = 1.0
    word_ngram: tuple[int, int] = (1, 2)
    char_ngram: tuple[int, int] = (2, 5)
    word_max: int = 40_000
    char_max: int = 30_000
    char_analyzer: str = "char_wb"
    min_df: int = 2

    def feature_key(self) -> tuple:
        return (
            self.use_char,
            self.word_ngram,
            self.char_ngram,
            self.word_max,
            self.char_max,
            self.char_analyzer,
            self.min_df,
        )


SPECS = (
    Spec(
        id="nb_word_char",
        kind="nb",
        features="tfidf word 1-2 + char_wb 2-5",
        hyperparameters="ComplementNB alpha=0.5 balanced sample_weight; thresholds 0.5",
    ),
    Spec(
        id="sgd_word_char",
        kind="sgd",
        features="tfidf word 1-2 + char_wb 2-5",
        hyperparameters="SGD log_loss alpha=1e-4 class_weight=balanced; thresholds 0.5",
    ),
    Spec(
        id="logreg_c4_word_char",
        kind="logreg",
        C=4.0,
        features="tfidf word 1-2 + char_wb 2-5",
        hyperparameters="LogReg C=4 class_weight=balanced max_iter=2000; thresholds 0.5",
    ),
    Spec(
        id="logreg_c025_word_char",
        kind="logreg",
        C=0.25,
        features="tfidf word 1-2 + char_wb 2-5",
        hyperparameters="LogReg C=0.25 class_weight=balanced max_iter=2000; thresholds 0.5",
    ),
    Spec(
        id="sgd_char36",
        kind="sgd",
        char_ngram=(3, 6),
        char_max=40_000,
        features="tfidf word 1-2 + char_wb 3-6",
        hyperparameters="SGD log_loss alpha=1e-4 class_weight=balanced; thresholds 0.5",
    ),
    Spec(
        id="nb_char36",
        kind="nb",
        char_ngram=(3, 6),
        char_max=40_000,
        features="tfidf word 1-2 + char_wb 3-6",
        hyperparameters="ComplementNB alpha=0.5 balanced sample_weight; thresholds 0.5",
    ),
    Spec(
        id="svc_mindf1",
        kind="svc",
        min_df=1,
        features="tfidf word 1-2 + char_wb 2-5 min_df=1",
        hyperparameters="Calibrated LinearSVC C=1 class_weight=balanced ensemble=False min_df=1; thresholds 0.5",
    ),
    Spec(
        id="svc_word3",
        kind="svc",
        word_ngram=(1, 3),
        features="tfidf word 1-3 + char_wb 2-5",
        hyperparameters="Calibrated LinearSVC C=1 class_weight=balanced ensemble=False word 1-3; thresholds 0.5",
    ),
    Spec(
        id="lgbm_word_char",
        kind="lgbm",
        features="tfidf word 1-2 + char_wb 2-5",
        hyperparameters=(
            "LGBM n_estimators=120 lr=0.1 num_leaves=31 class_weight=balanced "
            "colsample_bytree=0.3; thresholds 0.5"
        ),
    ),
)


def _indexes(rows: list[TicketRow]) -> tuple[list[int], list[int]]:
    train_index = [i for i, row in enumerate(rows) if row.split == "train"]
    val_index = [i for i, row in enumerate(rows) if row.split == "validation"]
    return train_index, val_index


def _matrices(
    spec: Spec, texts: list[str], train_index: list[int], val_index: list[int], cache: dict
):
    key = (spec.feature_key(), tuple(train_index), tuple(val_index))
    if key not in cache:
        print(f"  vectorizer {spec.feature_key()}", flush=True)
        vectorizer = make_vectorizer(
            spec.use_char,
            word_ngram=spec.word_ngram,
            char_ngram=spec.char_ngram,
            word_max=spec.word_max,
            char_max=spec.char_max,
            char_analyzer=spec.char_analyzer,
            min_df=spec.min_df,
        )
        x_train = vectorizer.fit_transform([texts[i] for i in train_index])
        x_val = vectorizer.transform([texts[i] for i in val_index])
        cache[key] = (vectorizer, x_train, x_val)
    return cache[key]


def _fit_eval(spec: Spec, rows, texts, train_index, val_index, cache, thresholds):
    _vectorizer, x_train, x_val = _matrices(spec, texts, train_index, val_index, cache)
    print(f"  heads {spec.id}", flush=True)
    category_model, secondary_models, urgent_model = fit_heads(
        x_train, train_index, rows, spec.kind, C=spec.C
    )
    category_proba, secondary_proba, urgent_proba = head_probabilities(
        category_model, secondary_models, urgent_model, x_val
    )
    predictions = decide(category_proba, secondary_proba, urgent_proba, thresholds)
    metrics = _metrics(rows, val_index, predictions, category_proba, urgent_proba)
    print(
        f"  {spec.id}: acc {metrics['accuracy']:.4f} macroF1 {metrics['macro_f1']:.4f} "
        f"sec {metrics['secondary_exact']:.4f} urgF1 {metrics['urgent_f1']:.4f}",
        flush=True,
    )
    bundle = {
        "category_model": category_model,
        "secondary_models": secondary_models,
        "urgent_model": urgent_model,
        "category_proba": category_proba,
        "secondary_proba": secondary_proba,
        "urgent_proba": urgent_proba,
        "predictions": predictions,
        "metrics": metrics,
    }
    return bundle


def _log(spec: Spec, metrics: dict, notes: str) -> None:
    append_experiment(
        {
            "id": spec.id,
            "branch": "classical",
            "features": spec.features,
            "hyperparameters": spec.hyperparameters,
            "stage_a_accuracy": metrics["accuracy"],
            "stage_a_macro_f1": metrics["macro_f1"],
            "stage_a_urgent_f1": metrics["urgent_f1"],
            "stage_a_secondary_exact": metrics["secondary_exact"],
            "notes": notes,
        }
    )


def _holdout_thresholds(spec: Spec, rows, texts, train_index):
    categories = [rows[i].category for i in train_index]
    positions = np.arange(len(train_index))
    fit_pos, hold_pos = train_test_split(
        positions, test_size=0.2, random_state=42, stratify=categories
    )
    fit_index = [train_index[int(i)] for i in fit_pos]
    hold_index = [train_index[int(i)] for i in hold_pos]
    print(f"  threshold holdout n={len(hold_index)}", flush=True)
    vectorizer = make_vectorizer(
        spec.use_char,
        word_ngram=spec.word_ngram,
        char_ngram=spec.char_ngram,
        word_max=spec.word_max,
        char_max=spec.char_max,
        char_analyzer=spec.char_analyzer,
        min_df=spec.min_df,
    )
    x_fit = vectorizer.fit_transform([texts[i] for i in fit_index])
    x_hold = vectorizer.transform([texts[i] for i in hold_index])
    category_model, secondary_models, urgent_model = fit_heads(
        x_fit, fit_index, rows, spec.kind, C=spec.C
    )
    if any(model is None for model in secondary_models.values()):
        raise RuntimeError("a secondary class was missing from the threshold holdout fit")
    category_proba, secondary_proba, urgent_proba = head_probabilities(
        category_model, secondary_models, urgent_model, x_hold
    )
    thresholds, point = select_thresholds(
        category_proba,
        secondary_proba,
        urgent_proba,
        [rows[i].secondary_category for i in hold_index],
        [rows[i].is_urgent for i in hold_index],
    )
    print(
        f"  thresholds secondary={thresholds['secondary']} urgent={thresholds['urgent']} "
        f"holdout sec {point.secondary_exact:.4f} urgF1 {point.urgent_f1:.4f}",
        flush=True,
    )
    return thresholds, point


def _save_candidate(spec: Spec, rows, texts, thresholds: dict) -> None:
    print(f"refitting {spec.id} on all {len(rows)} tickets", flush=True)
    vectorizer = make_vectorizer(
        spec.use_char,
        word_ngram=spec.word_ngram,
        char_ngram=spec.char_ngram,
        word_max=spec.word_max,
        char_max=spec.char_max,
        char_analyzer=spec.char_analyzer,
        min_df=spec.min_df,
    )
    matrix = vectorizer.fit_transform(texts)
    everything = list(range(len(rows)))
    category_model, secondary_models, urgent_model = fit_heads(
        matrix, everything, rows, spec.kind, C=spec.C
    )
    bundle = {
        "format": 1,
        "model_version": CANDIDATE_VERSION,
        "config_name": spec.id,
        "vectorizer": vectorizer,
        "category_model": category_model,
        "category_labels": list(CATEGORIES),
        "secondary_models": secondary_models,
        "secondary_labels": list(SECONDARY_CATEGORIES),
        "urgent_model": urgent_model,
        "thresholds": dict(thresholds),
        "trained_on": "train+validation",
        "n_rows": len(rows),
        "candidate": True,
    }
    CANDIDATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, CANDIDATE_PATH, compress=3)
    print(f"wrote {CANDIDATE_PATH}", flush=True)


def _write_decision(rows_out: list[str]) -> None:
    DECISION_PATH.write_text("\n".join(rows_out) + "\n", encoding="utf-8")
    print(f"wrote {DECISION_PATH}", flush=True)


def _baseline_spec() -> Spec:
    return Spec(
        id="svc_word_char",
        kind="svc",
        features="tfidf word 1-2 + char_wb 2-5",
        hyperparameters="Calibrated LinearSVC C=1 class_weight=balanced ensemble=False; thresholds 0.5",
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", help="run these spec ids and append rows, then stop")
    args = parser.parse_args()
    rows = load_all()
    texts = [build_input(row.channel, row.subject, row.text) for row in rows]
    train_index, val_index = _indexes(rows)
    cache: dict = {}
    if args.only:
        known = {spec.id for spec in SPECS}
        missing = sorted(set(args.only) - known)
        if missing:
            raise SystemExit(f"unknown spec: {', '.join(missing)}")
        for spec in SPECS:
            if spec.id not in args.only:
                continue
            print(f"== {spec.id}", flush=True)
            fitted = _fit_eval(spec, rows, texts, train_index, val_index, cache, THRESHOLDS)
            _log(
                spec,
                fitted["metrics"],
                "day-2 stage A; rules applied; thresholds 0.5; CV not re-run",
            )
            (REPORTS_DIR / f"{spec.id}.json").write_text(
                json.dumps({"id": spec.id, "stage_a": fitted["metrics"]}, indent=2) + "\n",
                encoding="utf-8",
            )
            delta = fitted["metrics"]["macro_f1"] - V0_MACRO_F1
            print(f"  delta macro-F1 vs v0 {delta:+.4f}", flush=True)
        return

    baseline = _baseline_spec()
    print("== baseline", flush=True)
    base = _fit_eval(baseline, rows, texts, train_index, val_index, cache, THRESHOLDS)
    fresh = base["metrics"]["macro_f1"]
    if abs(fresh - V0_MACRO_F1) > 0.002:
        raise SystemExit(
            f"stage A macro-F1 {fresh:.6f} does not match the logged v0 {V0_MACRO_F1:.6f}"
        )
    val_rows = [rows[i] for i in val_index]
    report = write_report(val_rows, base["predictions"], base["metrics"])
    print(f"wrote {report}", flush=True)

    results = [("svc_word_char", baseline, base)]
    for spec in SPECS:
        print(f"== {spec.id}", flush=True)
        fitted = _fit_eval(spec, rows, texts, train_index, val_index, cache, THRESHOLDS)
        _log(spec, fitted["metrics"], "day-2 stage A; rules applied; thresholds 0.5; CV not re-run")
        (REPORTS_DIR / f"{spec.id}.json").write_text(
            json.dumps({"id": spec.id, "stage_a": fitted["metrics"]}, indent=2) + "\n",
            encoding="utf-8",
        )
        results.append((spec.id, spec, fitted))

    winner_id, winner, winner_fit = max(results, key=lambda item: item[2]["metrics"]["macro_f1"])
    print(f"winner {winner_id} macroF1 {winner_fit['metrics']['macro_f1']:.4f}", flush=True)
    thresholds, holdout_point = _holdout_thresholds(winner, rows, texts, train_index)
    tuned_predictions = decide(
        winner_fit["category_proba"],
        winner_fit["secondary_proba"],
        winner_fit["urgent_proba"],
        thresholds,
    )
    tuned = _metrics(
        rows, val_index, tuned_predictions, winner_fit["category_proba"], winner_fit["urgent_proba"]
    )
    if abs(tuned["macro_f1"] - winner_fit["metrics"]["macro_f1"]) > 1e-6:
        raise SystemExit("thresholds changed category macro-F1; apply_rules must not do that")

    if (
        winner.id != "svc_word_char"
        or thresholds["secondary"] != 0.5
        or thresholds["urgent"] != 0.5
    ):
        tuned_spec = Spec(
            id=f"{winner.id}_tuned",
            kind=winner.kind,
            features=winner.features,
            hyperparameters=(
                f"{winner.hyperparameters}; frozen thresholds secondary={thresholds['secondary']} "
                f"urgent={thresholds['urgent']} from a 20% train holdout"
            ),
            use_char=winner.use_char,
            C=winner.C,
            word_ngram=winner.word_ngram,
            char_ngram=winner.char_ngram,
            word_max=winner.word_max,
            char_max=winner.char_max,
            char_analyzer=winner.char_analyzer,
            min_df=winner.min_df,
        )
        _log(
            tuned_spec,
            tuned,
            "day-2 v1 candidate operating point; thresholds not fit on validation; serving artifact unchanged",
        )
        _save_candidate(winner, rows, texts, thresholds)
        candidate_line = (
            f"This candidate is `artifacts/classical_v1.joblib`, version {CANDIDATE_VERSION}."
        )
    else:
        tuned_spec = winner
        candidate_line = (
            "No new artifact was written. `svc_word_char` at thresholds 0.5 remains the classical "
            "candidate, and it is already the serving file `artifacts/classical.joblib` (v0.1.0)."
        )

    lines = [
        "# Classical v1 candidate",
        "",
        "Stage A (fit on train, score on validation) chooses the model.",
        "Secondary and urgency thresholds are chosen on a stratified 20% holdout of train",
        "(seed 42), then frozen and applied once to the train-fit validation probabilities.",
        "The live serving file stays `artifacts/classical.joblib` at v0.1.0.",
        candidate_line,
        "Five-fold CV was not re-run: those scores are inflated by shared templates.",
        "",
        "| config | stage A acc | stage A macro-F1 | stage A urgent F1 | stage A secondary exact |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, _spec, fitted in results:
        metrics = fitted["metrics"]
        lines.append(
            f"| {name} | {metrics['accuracy']:.4f} | {metrics['macro_f1']:.4f} | "
            f"{metrics['urgent_f1']:.4f} | {metrics['secondary_exact']:.4f} |"
        )
    if tuned_spec.id != winner.id:
        lines.append(
            f"| {tuned_spec.id} | {tuned['accuracy']:.4f} | {tuned['macro_f1']:.4f} | "
            f"{tuned['urgent_f1']:.4f} | {tuned['secondary_exact']:.4f} |"
        )
    delta = tuned["macro_f1"] - fresh
    weak = sorted(tuned["per_class_f1"].items(), key=lambda item: item[1])[:4]
    weak_bits = ", ".join(f"{label} {score:.3f}" for label, score in weak)
    lines.extend(
        [
            "",
            f"Category model `{winner.id}` Stage A macro-F1 {winner_fit['metrics']['macro_f1']:.4f} "
            f"({delta:+.4f} versus the fresh v0 fit {fresh:.4f}).",
            f"Frozen thresholds: secondary {thresholds['secondary']}, urgent {thresholds['urgent']}, "
            f"review {thresholds['review']}.",
            f"Holdout at that point: secondary exact {holdout_point.secondary_exact:.4f}, "
            f"urgent F1 {holdout_point.urgent_f1:.4f}. "
            "A train holdout shares opening templates with the rest of train, so those holdout "
            "scores sit near 1.0 and are not used as the published number.",
            f"Validation at that point: secondary exact {tuned['secondary_exact']:.4f}, "
            f"urgent F1 {tuned['urgent_f1']:.4f} "
            f"(v0 urgent F1 {base['metrics']['urgent_f1']:.4f}, "
            f"v0 secondary exact {base['metrics']['secondary_exact']:.4f}).",
            f"Lowest class F1 on the candidate operating point: {weak_bits}.",
            "",
            "The candidate is not loaded by the API. Gate G1 still decides whether the encoder",
            "replaces classical serving. v0.1.0 remains what `/predict` runs.",
            "",
        ]
    )
    _write_decision(lines)


if __name__ == "__main__":
    main()
