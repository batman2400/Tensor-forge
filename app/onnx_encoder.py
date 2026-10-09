"""ONNX encoder branch. Runtime imports are tokenizers and onnxruntime only."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from ml.truncation import assemble_ids

OUTPUT_NAMES = ("category_logits", "secondary_logits", "urgent_logits")


def encode_arrays(
    raw_ids: list[int],
    cls_id: int,
    sep_id: int,
    pad_id: int,
    pad_to: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """One ticket as int64 arrays of shape (1, sequence)."""
    ids = assemble_ids(raw_ids, cls_id, sep_id)
    if pad_to is not None and len(ids) > pad_to:
        raise ValueError(f"sequence length {len(ids)} exceeds pad_to {pad_to}")
    if pad_to is not None and len(ids) < pad_to:
        pad_n = pad_to - len(ids)
        mask = [1] * len(ids) + [0] * pad_n
        ids = ids + [int(pad_id)] * pad_n
    else:
        mask = [1] * len(ids)
    return (
        np.asarray([ids], dtype=np.int64),
        np.asarray([mask], dtype=np.int64),
    )


def probabilities_from_logits(
    category: np.ndarray,
    secondary: np.ndarray,
    urgent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    category = np.asarray(category, dtype=np.float64)
    secondary = np.asarray(secondary, dtype=np.float64)
    urgent = np.asarray(urgent, dtype=np.float64)
    return _softmax(category), _sigmoid(secondary), _sigmoid(urgent).reshape(-1)


def measure_latency(encoder: OnnxEncoder, texts: list[str], warmup: int = 5) -> dict[str, float]:
    """Milliseconds per ticket at batch size 1. Warmup calls are not timed."""
    for text in texts[:warmup]:
        encoder.probabilities([text])
    samples: list[float] = []
    for text in texts:
        started = time.perf_counter()
        encoder.probabilities([text])
        samples.append((time.perf_counter() - started) * 1000.0)
    if not samples:
        return {"n": 0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    return {
        "n": float(len(samples)),
        "p50_ms": _percentile(samples, 0.50),
        "p95_ms": _percentile(samples, 0.95),
        "max_ms": max(samples),
    }


class OnnxEncoder:
    def __init__(
        self,
        session: Any,
        tokenizer: Any,
        cls_id: int,
        sep_id: int,
        pad_id: int,
        pad_to: int | None,
    ) -> None:
        self.session = session
        self.tokenizer = tokenizer
        self.cls_id = int(cls_id)
        self.sep_id = int(sep_id)
        self.pad_id = int(pad_id)
        self.pad_to = pad_to

    @classmethod
    def load(cls, model_path: Path, tokenizer_path: Path, meta: dict[str, Any]) -> OnnxEncoder:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        names = [item.name for item in session.get_outputs()]
        if tuple(names) != OUTPUT_NAMES:
            raise RuntimeError(f"encoder outputs {names} do not match {OUTPUT_NAMES}")
        pad_to = meta.get("pad_to")
        return cls(
            session,
            Tokenizer.from_file(str(tokenizer_path)),
            int(meta["cls_id"]),
            int(meta["sep_id"]),
            int(meta["pad_id"]),
            None if pad_to is None else int(pad_to),
        )

    def probabilities(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        categories: list[np.ndarray] = []
        secondaries: list[np.ndarray] = []
        urgents: list[float] = []
        for text in texts:
            raw = self.tokenizer.encode(text, add_special_tokens=False).ids
            input_ids, attention_mask = encode_arrays(
                list(raw), self.cls_id, self.sep_id, self.pad_id, self.pad_to
            )
            category, secondary, urgent = self.session.run(
                list(OUTPUT_NAMES),
                {"input_ids": input_ids, "attention_mask": attention_mask},
            )
            category_p, secondary_p, urgent_p = probabilities_from_logits(
                category, secondary, urgent
            )
            categories.append(
                np.asarray(category_p, dtype=np.float64).reshape(-1, category_p.shape[-1])[0]
            )
            secondaries.append(
                np.asarray(secondary_p, dtype=np.float64).reshape(-1, secondary_p.shape[-1])[0]
            )
            urgents.append(float(np.asarray(urgent_p, dtype=np.float64).reshape(-1)[0]))
        if not categories:
            return (
                np.zeros((0, 0), dtype=np.float64),
                np.zeros((0, 0), dtype=np.float64),
                np.zeros(0, dtype=np.float64),
            )
        return (
            np.stack(categories),
            np.stack(secondaries),
            np.asarray(urgents, dtype=np.float64),
        )


def _softmax(logits: np.ndarray) -> np.ndarray:
    flat = np.atleast_2d(logits.reshape(-1, logits.shape[-1]))
    shifted = flat - flat.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac
