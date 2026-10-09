"""Text normalization shared by training and serving.

Unicode NFC, keep ZWJ/ZWNJ (Sinhala and Tamil need them), strip other control
and format characters, collapse whitespace, lowercase Latin letters only.
"""

from __future__ import annotations

import re
import unicodedata

_WHITESPACE = re.compile(r"\s+")
# Cf characters we must not strip. ZWJ (U+200D) builds Sinhala conjuncts; ZWNJ (U+200C)
# is used in Tamil and Sinhala to block a conjunct.
_KEEP_FORMAT = frozenset("\u200c\u200d")


def _is_dropped(ch: str) -> bool:
    if ch in _KEEP_FORMAT:
        return False
    # Newlines and tabs are whitespace, not noise: they collapse to a space later.
    if ch in "\n\r\t":
        return False
    category = unicodedata.category(ch)
    return category in {"Cc", "Cf", "Cs"}


def _lower_latin(ch: str) -> str:
    if unicodedata.category(ch) != "Lu":
        return ch
    if unicodedata.name(ch, "").startswith("LATIN"):
        return ch.lower()
    return ch


def normalize(text: str) -> str:
    """Normalize one text field. Does not add the channel/subject wrapper."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    cleaned = "".join(_lower_latin(ch) for ch in text if not _is_dropped(ch))
    return _WHITESPACE.sub(" ", cleaned).strip()


def build_input(channel: str, subject: str | None, text: str) -> str:
    """Model input string. `ticket_id` and `language` are intentionally not arguments."""
    body = normalize(text)
    subj = normalize(subject or "")
    if subj:
        return f"[{channel}] {subj} || {body}"
    return f"[{channel}] || {body}"


# Politeness, pointers, and thread prefixes. A body made only of these names no issue.
# Category words are intentionally absent: "refund" alone is a real complaint.
_FILLER = frozenset(
    {
        "a",
        "an",
        "the",
        "i",
        "me",
        "my",
        "you",
        "your",
        "we",
        "us",
        "please",
        "pls",
        "plz",
        "kindly",
        "help",
        "assist",
        "assistance",
        "see",
        "below",
        "above",
        "attached",
        "attachment",
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank",
        "thankyou",
        "urgent",
        "asap",
        "this",
        "that",
        "it",
        "with",
        "for",
        "and",
        "to",
        "of",
        "in",
        "on",
        "regarding",
        "about",
        "re",
        "fw",
        "fwd",
        "need",
        "needed",
        "want",
        "wanted",
        "issue",
        "problem",
        "query",
        "question",
        "can",
        "could",
        "would",
        "eh",
        "leh",
        "lah",
        "lor",
        "කරුණාකර",
        "උදව්",
        "தயவுசெய்து",
        "உதவி",
    }
)
# ASCII punctuation only. Sinhala virama and Tamil marks are letters here, not edges.
_EDGE_PUNCT = ".,!?;:\"'()[]{}<>/\\|_-~`*"


def is_vague_body(text: str | None) -> bool:
    """True when the field does not name an issue.

    "please help" and "see below" are vague. "refund" and a Sinhala sentence are not.
    The same check is used for a subject, so a bare "Re:" does not count as a conflict.
    """
    tokens = []
    for raw in normalize(text or "").split(" "):
        token = raw.strip(_EDGE_PUNCT)
        if token:
            tokens.append(token)
    return not tokens or all(token in _FILLER for token in tokens)
