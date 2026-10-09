from __future__ import annotations

from app.text import build_input, is_vague_body, normalize


def test_lowercases_latin_only():
    assert normalize("ABC") == "abc"
    assert normalize("É") == "é"
    assert normalize("Ω") == "Ω"
    assert normalize("Hello அ") == "hello அ"


def test_keeps_zwj_and_strips_other_controls():
    conjunct = "\u0d9a\u0dca\u200d\u0dbb"
    assert "\u200d" in normalize(conjunct)
    assert "\u200c" in normalize("a\u200cb")
    assert normalize("a\x00b") == "ab"
    assert normalize("a\n\nb") == "a b"
    assert normalize("  a   b  ") == "a b"


def test_keeps_emoji_and_digits():
    assert normalize("OK 🙏 12") == "ok 🙏 12"


def test_vague_body_is_only_filler():
    assert is_vague_body("please help")
    assert is_vague_body("See below")
    assert is_vague_body("Re:")
    assert is_vague_body("කරුණාකර උදව්")
    assert is_vague_body("")
    assert not is_vague_body("refund")
    assert not is_vague_body("please refund the extra charge")
    assert not is_vague_body("මගේ order එක නැත")


def test_build_input_shape():
    assert build_input("email", "Refund", "Please") == "[email] refund || please"
    assert build_input("chat", "", "Hello") == "[chat] || hello"
    assert build_input("chat", None, "Hello") == "[chat] || hello"
    assert build_input("chat", "   ", "Hello") == "[chat] || hello"
