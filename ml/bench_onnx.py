"""Time an exported encoder without PyTorch.

python -m ml.bench_onnx --model ml/runs/encoder_onnx_smoke/encoder.int8.onnx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.onnx_encoder import OnnxEncoder, measure_latency
from app.text import build_input


def _texts(data_dir: Path, limit: int) -> list[str]:
    if (
        not (data_dir / "validation.csv").is_file()
        and not (data_dir / "validation.jsonl").is_file()
    ):
        return []
    from ml.data import load_all

    rows = [row for row in load_all(data_dir) if row.split == "validation"]
    return [build_input(row.channel, row.subject, row.text) for row in rows[:limit]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark an exported ONNX encoder.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--meta", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--tickets", type=int, default=50)
    args = parser.parse_args()
    model = args.model
    tokenizer = args.tokenizer or model.with_name("tokenizer.json")
    meta_path = args.meta or model.with_name("encoder_meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    encoder = OnnxEncoder.load(model, tokenizer, meta)
    from ml.export_onnx import _rss_bytes

    print(f"rss_after_load_mb={_rss_bytes() / 1e6:.1f}", flush=True)
    texts = _texts(args.data_dir, args.tickets)
    if len(texts) < args.tickets:
        texts = texts + [build_input("chat", "hello", "where is my order")] * (
            args.tickets - len(texts)
        )
    latency = measure_latency(encoder, texts[: args.tickets], warmup=5)
    long_text = build_input("email", "subject", "word " * 2000)
    long_latency = measure_latency(encoder, [long_text] * 20, warmup=2)
    print(
        f"validation n={int(latency['n'])} p50={latency['p50_ms']:.1f} "
        f"p95={latency['p95_ms']:.1f} max={latency['max_ms']:.1f}",
        flush=True,
    )
    print(
        f"long n={int(long_latency['n'])} p50={long_latency['p50_ms']:.1f} "
        f"p95={long_latency['p95_ms']:.1f} max={long_latency['max_ms']:.1f}",
        flush=True,
    )
    print(f"rss_after_bench_mb={_rss_bytes() / 1e6:.1f}", flush=True)


if __name__ == "__main__":
    main()
