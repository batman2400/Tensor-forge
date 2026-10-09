"""Stage-A error slices for the classical baseline.

The slices follow the labelling rules in ``data/DATA_NOTES.md``. The report
counts mistakes. It does not quote ticket text.

    python -m ml.error_analysis
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from ml.data import TicketRow
from ml.evaluate import REPORTS_DIR

FINDINGS_PATH = REPORTS_DIR / "error_analysis.md"

# Same surface patterns as ml/eda.py. They tag slices; they are not features.
_INJECTION = re.compile(
    r"ignore (all |previous |the above)|disregard (all |previous )|system prompt|"
    r"you are (a |an )?(assistant|bot)|mark this (as )?urgent",
    re.IGNORECASE,
)
_RESOLVED = re.compile(
    r"\b(resolved|already refunded|no longer|sorted out|this was fixed|issue is closed)\b",
    re.IGNORECASE,
)


def _is_quoted(text: str) -> bool:
    lowered = text.lower()
    if "wrote:" in lowered or "original message" in lowered:
        return True
    return any(line.strip().startswith(">") for line in text.splitlines())


def _mask(rows: list[TicketRow], keep) -> list[bool]:
    return [keep(row) for row in rows]


def rule_masks(rows: list[TicketRow]) -> dict[str, list[bool]]:
    """One boolean per validation row, for each labelling-rule slice."""
    return {
        "safety_conduct": _mask(rows, lambda row: row.category == "safety_conduct"),
        "late_food_refund": _mask(
            rows,
            lambda row: (
                row.category == "delivery_delay" and row.secondary_category == "payment_refund"
            ),
        ),
        "missing_food_refund": _mask(
            rows,
            lambda row: (
                row.category == "order_missing_wrong" and row.secondary_category == "payment_refund"
            ),
        ),
        "lost_item": _mask(rows, lambda row: row.category == "lost_item"),
        "account_promo": _mask(rows, lambda row: row.category == "account_promo"),
        "app_technical": _mask(rows, lambda row: row.category == "app_technical"),
        "general_inquiry": _mask(rows, lambda row: row.category == "general_inquiry"),
        "nonurgent_safety": _mask(
            rows, lambda row: row.category == "safety_conduct" and not row.is_urgent
        ),
        "spam_irrelevant": _mask(rows, lambda row: row.category == "spam_irrelevant"),
        "injection_text": _mask(rows, lambda row: _INJECTION.search(row.text) is not None),
        "resolved_wording": _mask(rows, lambda row: _RESOLVED.search(row.text) is not None),
        "quoted_correspondence": _mask(rows, lambda row: _is_quoted(row.text)),
    }


def _rate(flags: list[bool]) -> float:
    if not flags:
        return 0.0
    return sum(flags) / len(flags)


def slice_summary(
    rows: list[TicketRow],
    predictions: list[dict],
    mask: list[bool],
) -> dict:
    chosen = [(row, pred) for row, pred, keep in zip(rows, predictions, mask, strict=True) if keep]
    n = len(chosen)
    if n == 0:
        return {
            "n": 0,
            "accuracy": 0.0,
            "secondary_exact": 0.0,
            "urgent_false_positive_rate": 0.0,
            "mistakes": [],
        }
    primary_hits = [row.category == pred["category"] for row, pred in chosen]
    secondary_hits = [row.secondary_category == pred["secondary_category"] for row, pred in chosen]
    urgent_fp = [pred["is_urgent"] and not row.is_urgent for row, pred in chosen]
    mistakes = Counter(pred["category"] for row, pred in chosen if row.category != pred["category"])
    return {
        "n": n,
        "accuracy": _rate(primary_hits),
        "secondary_exact": _rate(secondary_hits),
        "urgent_false_positive_rate": _rate(urgent_fp),
        "mistakes": mistakes.most_common(3),
    }


def confusion_pairs(rows: list[TicketRow], predictions: list[dict], limit: int = 8) -> list[tuple]:
    counts = Counter(
        (row.category, pred["category"])
        for row, pred in zip(rows, predictions, strict=True)
        if row.category != pred["category"]
    )
    return counts.most_common(limit)


def _fmt_mistakes(mistakes: list[tuple]) -> str:
    if not mistakes:
        return "none"
    return ", ".join(f"{label} {count}" for label, count in mistakes)


def _fmt_slice_row(name: str, summary: dict) -> str:
    return (
        f"| {name} | {summary['n']} | {summary['accuracy']:.3f} | "
        f"{summary['secondary_exact']:.3f} | {summary['urgent_false_positive_rate']:.3f} | "
        f"{_fmt_mistakes(summary['mistakes'])} |"
    )


def render_report(
    rows: list[TicketRow],
    predictions: list[dict],
    metrics: dict,
) -> str:
    """Markdown for the Stage A baseline. `rows` and `predictions` are validation only."""
    masks = rule_masks(rows)
    slices = {name: slice_summary(rows, predictions, mask) for name, mask in masks.items()}
    false_safety = sum(
        row.category != "safety_conduct" and pred["category"] == "safety_conduct"
        for row, pred in zip(rows, predictions, strict=True)
    )
    lost_in = sum(
        row.category != "lost_item" and pred["category"] == "lost_item"
        for row, pred in zip(rows, predictions, strict=True)
    )
    promo_as_app = sum(
        row.category == "account_promo" and pred["category"] == "app_technical"
        for row, pred in zip(rows, predictions, strict=True)
    )
    app_as_promo = sum(
        row.category == "app_technical" and pred["category"] == "account_promo"
        for row, pred in zip(rows, predictions, strict=True)
    )
    pairs = confusion_pairs(rows, predictions)
    per_class = sorted(metrics["per_class_f1"].items(), key=lambda item: item[1])
    class_lines = "\n".join(
        f"| {label} | {score:.3f} | {metrics['per_class_support'][label]} |"
        for label, score in per_class
    )
    pair_lines = "\n".join(f"| {gold} | {pred} | {count} |" for (gold, pred), count in pairs)
    slice_lines = "\n".join(_fmt_slice_row(name, slices[name]) for name in masks)
    language_lines = "\n".join(
        f"| {language} | {stats['n']} | {stats['accuracy']:.3f} | {stats['macro_f1']:.3f} |"
        for language, stats in metrics["by_language"].items()
    )
    channel_lines = "\n".join(
        f"| {channel} | {stats['n']} | {stats['accuracy']:.3f} | {stats['macro_f1']:.3f} |"
        for channel, stats in metrics["by_channel"].items()
    )
    overall_fp = sum(
        pred["is_urgent"] and not row.is_urgent for row, pred in zip(rows, predictions, strict=True)
    )
    overall_fp_rate = overall_fp / len(rows) if rows else 0.0
    return "\n".join(
        [
            "# Error analysis (classical v0, Stage A)",
            "",
            "Fit `svc_word_char` on train, score the 800 validation tickets, thresholds 0.5,",
            "post-processing rules on. This is the same setup as the day-1 serving baseline.",
            "Counts only: no ticket text.",
            "",
            f"Stage A accuracy {metrics['accuracy']:.4f}, macro-F1 {metrics['macro_f1']:.4f},",
            f"urgent F1 {metrics['urgent_f1']:.4f} (precision {metrics['urgent_precision']:.4f},",
            f"recall {metrics['urgent_recall']:.4f}), secondary exact {metrics['secondary_exact']:.4f}.",
            f"Urgent false positives: {overall_fp} / {len(rows)} ({overall_fp_rate:.3f}).",
            "",
            "## Where category errors concentrate",
            "",
            "Macro-F1 is pulled down by the food-and-money cluster. `order_missing_wrong` is the",
            "weak class. `lost_item` looks better on recall than precision because other classes",
            f"fall into it ({lost_in} validation tickets predicted `lost_item` with a different gold label).",
            "Ride tickets are the main source of false `safety_conduct`",
            f"({false_safety} non-safety tickets predicted as safety).",
            "",
            "| category | F1 | support |",
            "| --- | --- | --- |",
            class_lines,
            "",
            "Largest gold → predicted pairs:",
            "",
            "| gold | predicted | count |",
            "| --- | --- | --- |",
            pair_lines,
            "",
            "## Labelling-rule slices",
            "",
            "Accuracy is the primary category on that slice. Secondary exact is the full pair,",
            "including a correct null. Urgent false-positive rate is predicted urgent when the",
            "gold label is not. Mistake lists are the predicted category when the primary is wrong.",
            "",
            "| slice | n | accuracy | secondary exact | urgent FP rate | top wrong predictions |",
            "| --- | --- | --- | --- | --- | --- |",
            slice_lines,
            "",
            f"OTP/login versus post-login, measured as the two-way confusion: "
            f"{promo_as_app} `account_promo` predicted `app_technical`, "
            f"{app_as_promo} `app_technical` predicted `account_promo`.",
            "Late-food refunds are `delivery_delay` with secondary `payment_refund`.",
            "Missing-food refunds are `order_missing_wrong` with secondary `payment_refund`.",
            "Non-urgent safety is the measurable stand-in for a resolved historical safety complaint:",
            "the gold label is safety and not urgent. Spam is a separate row and should stay exact,",
            "because the serving rule clears urgency and secondary after a spam primary.",
            "",
            "Injection, resolved wording, and quoted correspondence use the same surface patterns",
            "as the EDA counts. They are slices, not keyword features. A higher urgent false-positive",
            "rate on the injection slice than the overall rate means the model is treating the",
            "instruction text as a signal.",
            "",
            "## Language and channel",
            "",
            "Singlish is much easier than Sinhala or English on this split. Shared opening templates",
            "are a known property of the pool, so a high Singlish score is not a reason to add",
            "`language` as a feature. The model still does not receive `language` or `ticket_id`.",
            "",
            "| language | n | accuracy | macro-F1 |",
            "| --- | --- | --- | --- |",
            language_lines,
            "",
            "| channel | n | accuracy | macro-F1 |",
            "| --- | --- | --- | --- |",
            channel_lines,
            "",
            "Email has subjects; chat and calls do not. That is already in `build_input`.",
            "",
            "## What this asks of the v1 candidate",
            "",
            "- Lift `order_missing_wrong` away from `lost_item` and `payment_refund` without giving",
            "  those classes away. That is the macro-F1 lever.",
            "- Stop calling ordinary ride issues `safety_conduct`.",
            "- Keep secondary exact near the v0 level (0.955). It is already a strength.",
            "- Urgency is high-recall and lower-precision. Non-urgent safety and resolved wording",
            "  are where false urgency shows up. The injection slice did not raise urgent false",
            "  positives on this validation split, so there is no evidence here for a special",
            "  injection parser.",
            "- `app_technical` is often predicted as `order_missing_wrong`. The OTP versus",
            "  post-login swap is small by comparison.",
            "- Do not add a hard keyword rule for refunds, OTP, or resolved mail. The native-script",
            "  rows will not match an English keyword, and the slices are not precise enough.",
            "",
        ]
    )


def write_report(
    rows: list[TicketRow], predictions: list[dict], metrics: dict, path: Path | None = None
) -> Path:
    destination = path or FINDINGS_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_report(rows, predictions, metrics), encoding="utf-8")
    return destination


def main() -> None:
    """Fit the v0 configuration on train and write the validation report."""
    from app.text import build_input
    from ml.data import load_all
    from ml.train_classical import THRESHOLDS, fit_heads, make_vectorizer, predict_matrix

    rows = load_all()
    texts = [build_input(row.channel, row.subject, row.text) for row in rows]
    train_index = [i for i, row in enumerate(rows) if row.split == "train"]
    val_index = [i for i, row in enumerate(rows) if row.split == "validation"]
    vectorizer = make_vectorizer(True)
    x_train = vectorizer.fit_transform([texts[i] for i in train_index])
    x_val = vectorizer.transform([texts[i] for i in val_index])
    heads = fit_heads(x_train, train_index, rows, "svc")
    predictions, category_proba, urgent_proba = predict_matrix(*heads, x_val, THRESHOLDS)
    from ml.train_classical import _metrics

    metrics = _metrics(rows, val_index, predictions, category_proba, urgent_proba)
    val_rows = [rows[i] for i in val_index]
    path = write_report(val_rows, predictions, metrics)
    print(f"wrote {path}", flush=True)
    print(
        f"stage A macro-F1 {metrics['macro_f1']:.4f} acc {metrics['accuracy']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
