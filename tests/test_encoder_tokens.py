"""Head-and-tail truncation. No model download."""

from app.onnx_encoder import encode_arrays
from ml.truncation import HEAD_TOKENS, MAX_LENGTH, TAIL_TOKENS, assemble_ids, head_tail_window


def test_short_sequence_is_unchanged():
    ids = list(range(10))
    assert head_tail_window(ids, MAX_LENGTH - 2, HEAD_TOKENS, TAIL_TOKENS) == ids


def test_long_sequence_keeps_head_and_tail():
    ids = list(range(1000))
    window = head_tail_window(ids, MAX_LENGTH - 2, HEAD_TOKENS, TAIL_TOKENS)
    assert len(window) == HEAD_TOKENS + TAIL_TOKENS
    assert window[:HEAD_TOKENS] == ids[:HEAD_TOKENS]
    assert window[-TAIL_TOKENS:] == ids[-TAIL_TOKENS:]


def test_tail_shrinks_when_the_budget_is_tight():
    ids = list(range(50))
    window = head_tail_window(ids, 10, head=8, tail=8)
    assert window == ids[:8] + ids[-2:]


def test_assemble_ids_wraps_special_tokens():
    assert assemble_ids([5, 6, 7], cls_id=0, sep_id=2) == [0, 5, 6, 7, 2]


def test_assemble_ids_truncates_to_max_length():
    ids = assemble_ids(list(range(1000)), cls_id=0, sep_id=2)
    assert len(ids) == MAX_LENGTH
    assert ids[0] == 0
    assert ids[-1] == 2
    assert ids[1 : 1 + HEAD_TOKENS] == list(range(HEAD_TOKENS))
    assert ids[-1 - TAIL_TOKENS : -1] == list(range(1000 - TAIL_TOKENS, 1000))


def test_encode_arrays_pads_when_asked():
    ids, mask = encode_arrays([7, 8], cls_id=1, sep_id=2, pad_id=0, pad_to=6)
    assert ids.dtype == mask.dtype
    assert ids.tolist() == [[1, 7, 8, 2, 0, 0]]
    assert mask.tolist() == [[1, 1, 1, 1, 0, 0]]


def test_encode_arrays_stays_tight_without_padding():
    ids, mask = encode_arrays([7], cls_id=1, sep_id=2, pad_id=0, pad_to=None)
    assert ids.tolist() == [[1, 7, 2]]
    assert mask.tolist() == [[1, 1, 1]]
