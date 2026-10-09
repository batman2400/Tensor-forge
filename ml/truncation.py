"""Head-and-tail token window shared by encoder training and, later, serving.

A 256-token budget keeps the start of the ticket (channel, subject, opening) and
the end (the ask is often there). The middle is what gets dropped on long texts.
"""

from __future__ import annotations

MAX_LENGTH = 256
# Content tokens only. CLS and SEP use the other two positions (190 + 64 + 2 = 256).
HEAD_TOKENS = 190
TAIL_TOKENS = 64


def head_tail_window(token_ids: list[int], content_budget: int, head: int, tail: int) -> list[int]:
    """Keep the first `head` and last `tail` ids when the sequence exceeds the budget."""
    if content_budget <= 0:
        return []
    if len(token_ids) <= content_budget:
        return list(token_ids)
    head_n = min(max(head, 0), content_budget)
    tail_n = min(max(tail, 0), content_budget - head_n)
    if tail_n <= 0:
        return list(token_ids[:content_budget])
    return list(token_ids[:head_n]) + list(token_ids[-tail_n:])


def assemble_ids(raw_ids: list[int], cls_id: int, sep_id: int) -> list[int]:
    """Content window plus the CLS and SEP ids. Length is at most `MAX_LENGTH`."""
    window = head_tail_window(list(raw_ids), MAX_LENGTH - 2, HEAD_TOKENS, TAIL_TOKENS)
    return [int(cls_id), *[int(token_id) for token_id in window], int(sep_id)]
