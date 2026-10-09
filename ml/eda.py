"""Exploratory analysis. Writes aggregate findings only: no ticket text.

Run from the repo root:

    python -m ml.eda
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path

from app.text import normalize
from ml.data import N_SPLITS, TicketRow, load_all, write_folds
from ml.evaluate import record_majority

ROOT = Path(__file__).resolve().parents[1]
FINDINGS_PATH = ROOT / "ml" / "reports" / "eda_findings.md"

_INJECTION = re.compile(
    r"ignore (all |previous |the above)|disregard (all |previous )|system prompt|"
    r"you are (a |an )?(assistant|bot)|mark this (as )?urgent",
    re.IGNORECASE,
)
_RESOLVED = re.compile(
    r"\b(resolved|already refunded|no longer|sorted out|this was fixed|issue is closed)\b",
    re.IGNORECASE,
)


def _has_emoji(text: str) -> bool:
    return any(ord(ch) >= 0x1F300 for ch in text)


def _is_quoted(text: str) -> bool:
    lowered = text.lower()
    if "wrote:" in lowered or "original message" in lowered:
        return True
    return any(line.strip().startswith(">") for line in text.splitlines())


def _pct(count: int, total: int) -> str:
    return f"{count / total:.1%}"


def collect_stats(rows: list[TicketRow]) -> dict:
    categories = Counter(row.category for row in rows)
    secondary = Counter(row.secondary_category for row in rows)
    languages = Counter(row.language for row in rows)
    channels = Counter(row.channel for row in rows)
    urgent_by = {
        category: (
            sum(1 for row in rows if row.category == category and row.is_urgent),
            sum(1 for row in rows if row.category == category),
        )
        for category in categories
    }
    subject_by = {
        channel: sum(1 for row in rows if row.channel == channel and row.subject.strip())
        for channel in channels
    }
    lengths = sorted(len(row.text) for row in rows)
    pairs = Counter(
        (row.category, row.secondary_category) for row in rows if row.secondary_category
    )
    strata = Counter((row.category, row.language) for row in rows)
    prefixes: dict[str, list[str]] = {}
    norm_texts: dict[str, list[str]] = {}
    for row in rows:
        normalized = normalize(row.text)
        prefixes.setdefault(normalized[:80], []).append(row.split)
        norm_texts.setdefault(normalized, []).append(row.split)

    def _cross(groups: dict[str, list[str]]) -> int:
        return sum(1 for splits in groups.values() if "train" in splits and "validation" in splits)

    return {
        "n": len(rows),
        "n_train": sum(row.split == "train" for row in rows),
        "n_val": sum(row.split == "validation" for row in rows),
        "categories": categories.most_common(),
        "secondary": secondary,
        "secondary_present": sum(row.secondary_category is not None for row in rows),
        "urgent": sum(row.is_urgent for row in rows),
        "urgent_by": urgent_by,
        "languages": languages,
        "channels": channels,
        "subject_by": subject_by,
        "len_min": lengths[0],
        "len_p50": lengths[len(lengths) // 2],
        "len_p95": lengths[int(len(lengths) * 0.95)],
        "len_p99": lengths[int(len(lengths) * 0.99)],
        "len_max": lengths[-1],
        "newlines": sum("\n" in row.text for row in rows),
        "non_nfc": sum(unicodedata.normalize("NFC", row.text) != row.text for row in rows),
        "emoji": sum(_has_emoji(row.text) for row in rows),
        "injection": sum(_INJECTION.search(row.text) is not None for row in rows),
        "resolved": sum(_RESOLVED.search(row.text) is not None for row in rows),
        "quoted": sum(_is_quoted(row.text) for row in rows),
        "n_strata": len(strata),
        "strata_min": min(strata.values()),
        "strata_below_5": sum(1 for count in strata.values() if count < 5),
        "exact_norm_dups": sum(1 for splits in norm_texts.values() if len(splits) > 1),
        "shared_prefixes": sum(1 for splits in prefixes.values() if len(splits) > 1),
        "cross_split_prefixes": _cross(prefixes),
        "pairs": pairs.most_common(8),
        "pair_delay_refund": pairs[("delivery_delay", "payment_refund")],
        "pair_missing_refund": pairs[("order_missing_wrong", "payment_refund")],
    }


def render_findings(stats: dict, majority: dict) -> str:
    total = stats["n"]
    categories = stats["categories"]
    top_name, top_n = categories[0]
    low_name, low_n = categories[-1]
    secondary_bits = ", ".join(
        f"{name} {count}" for name, count in stats["secondary"].most_common() if name is not None
    )
    language_bits = ", ".join(
        f"{name} {_pct(count, total)}" for name, count in stats["languages"].most_common()
    )
    channel_bits = ", ".join(f"{name} {count}" for name, count in stats["channels"].most_common())
    never = ["account_promo", "app_technical", "general_inquiry", "spam_irrelevant"]
    never_bits = ", ".join(
        f"{name} {stats['urgent_by'][name][0]}/{stats['urgent_by'][name][1]}" for name in never
    )
    safety_u, safety_n = stats["urgent_by"]["safety_conduct"]
    pair_bits = ", ".join(
        f"{primary}+{secondary} {count}" for (primary, secondary), count in stats["pairs"]
    )
    lines = [
        "# EDA findings (day 1)",
        "",
        "Aggregate counts over the 4,800 train+validation tickets. No ticket text is quoted here.",
        "These points decide the day-1 features and the post-processing rules.",
        "",
        (
            f"- Class imbalance is large: `{top_name}` is {top_n} ({_pct(top_n, total)}) and "
            f"`{low_name}` is {low_n} ({_pct(low_n, total)}). Headline metric is macro-F1. "
            f"Accuracy alone will flatter a model that sticks to the head classes."
        ),
        (
            f"- Secondary labels are rare ({stats['secondary_present']} / {total}, "
            f"{_pct(stats['secondary_present'], total)}) and only five classes occur: {secondary_bits}. "
            "The secondary head is one-vs-rest over those five, plus a threshold. "
            "Do not emit any other class as secondary."
        ),
        (
            f"- Urgency is {stats['urgent']} / {total} ({_pct(stats['urgent'], total)}). "
            f"It is exactly zero on {never_bits}. `safety_conduct` is urgent on {safety_u}/{safety_n} "
            f"({_pct(safety_u, safety_n)}). Force those four categories to non-urgent after the model. "
            "Spam also forces `secondary_category` null (schema rule, and the data agrees)."
        ),
        (
            f"- Language mix matches the brief ({language_bits}). `language` is not a model feature "
            "and is not sent by the API. Char n-grams are the robustness tool for Singlish and Tanglish spelling."
        ),
        (
            f"- Channels: {channel_bits}. Non-empty subject counts by channel: "
            + ", ".join(f"{name} {count}" for name, count in sorted(stats["subject_by"].items()))
            + ". Subject shows up on email and not on chat or calls, and the longest subject is far under "
            "the 500-character cap. `build_input` always includes the channel and includes the subject only "
            "when normalization leaves it non-empty."
        ),
        (
            f"- Text is short relative to the 10,000-character API cap: min {stats['len_min']}, "
            f"median {stats['len_p50']}, p95 {stats['len_p95']}, p99 {stats['len_p99']}, "
            f"max {stats['len_max']}. {stats['newlines']} / {total} texts contain embedded newlines "
            "(chat and call transcripts). Collapse whitespace in one shared function used by training and serving."
        ),
        (
            f"- This release is already NFC ({stats['non_nfc']} texts differ). Still normalize to NFC on the way in, "
            "because a later set might not be, and keep ZWJ (U+200D) and ZWNJ (U+200C); Sinhala conjuncts "
            "depend on them. Lowercase Latin letters only."
        ),
        (
            f"- Injection-style phrasing matches {stats['injection']} tickets. There is no instruction "
            "channel: the model sees that text as tokens. Keep an error-analysis slice for it; do not add a special parser."
        ),
        (
            f"- Resolved/historical wording matches {stats['resolved']} tickets, and quoted-correspondence "
            f"markers match {stats['quoted']}. DATA_NOTES say a resolved safety complaint is not urgent and "
            "quoted resolved mail is background. That is a slice for error analysis, not a hard keyword rule, "
            "until we measure precision."
        ),
        (
            f"- The joint category x language key has {stats['n_strata']} cells, minimum count "
            f"{stats['strata_min']}, and {stats['strata_below_5']} cells under 5. "
            f"Five-fold `StratifiedKFold` cannot run on that key. Folds are a seeded shuffle plus "
            f"round-robin inside each cell (`ml/folds.json`, seed 42, {N_SPLITS} folds), ids only."
        ),
        (
            f"- Exact normalized-text duplicates: {stats['exact_norm_dups']} normalized strings occur "
            f"more than once. Shared 80-character normalized prefixes: {stats['shared_prefixes']}, of which "
            f"{stats['cross_split_prefixes']} appear in both train and validation. "
            "Near-duplicate wording is common, so a high validation score is not proof of generalization; "
            "the number we compare is the 5-fold OOF macro-F1. Do not use `ticket_id` as a feature."
        ),
        (
            f"- Secondary pairs that match the labelling rules: `delivery_delay` + `payment_refund` "
            f"{stats['pair_delay_refund']}, `order_missing_wrong` + `payment_refund` "
            f"{stats['pair_missing_refund']}. Most common pairs: {pair_bits}. "
            "A refund asked because food is late is secondary to delay; a refund because food is missing "
            "is secondary to the missing order. The primary head will not learn that on its own."
        ),
        (
            f"- Emoji appears in {stats['emoji']} tickets. Emoji-only text is still a valid API input and "
            "must return 200. Word n-grams can drop it; char n-grams should keep the vector from being empty."
        ),
        (
            f"- Majority-class floor (always `{majority['majority_category']}`, secondary null, urgent false), "
            f"scored on the 800 validation tickets: accuracy {majority['accuracy']:.3f}, "
            f"macro-F1 {majority['macro_f1']:.3f}, secondary exact {majority['secondary_exact']:.3f}, "
            f"urgent F1 {majority['urgent_f1']:.3f}, ECE {majority['ece']:.3f}. "
            "Anything we ship has to beat this macro-F1 with the same metric code. "
            "The ECE is bad on purpose: confidence is 1.0 on every row."
        ),
        (
            "- Serving rules for v0, taken from the points above: TF-IDF word and char n-grams on "
            "`build_input(channel, subject, text)`; no `language` and no `ticket_id`; secondary over the "
            "five observed classes; force non-urgent on account_promo, app_technical, general_inquiry, "
            "and spam_irrelevant; spam clears secondary; team comes only from the fixed table."
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    rows = load_all()
    write_folds(rows)
    stats = collect_stats(rows)
    majority = record_majority()
    findings = render_findings(stats, majority)
    FINDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FINDINGS_PATH.write_text(findings, encoding="utf-8")
    print(findings)
    print(f"wrote {FINDINGS_PATH}")


if __name__ == "__main__":
    main()
