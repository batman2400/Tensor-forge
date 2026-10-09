"""Export the encoder to one int8 ONNX graph and check it against PyTorch.

The default checkpoint is the laptop smoke run. It is the same
``multilingual-e5-small`` graph as the Kaggle model, so its latency is the
graph's latency. The smoke head weights are not the serving model, and this
command does not replace ``artifacts/classical.joblib``.

    python -m ml.export_onnx
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "ml" / "runs" / "encoder_smoke" / "stage_a" / "model.pt"
DEFAULT_OUT = ROOT / "ml" / "runs" / "encoder_onnx_smoke"
REPORT_JSON = ROOT / "ml" / "reports" / "onnx_latency.json"
REPORT_MD = ROOT / "ml" / "reports" / "onnx_latency.md"
FP32_MAX_ABS = 1e-3
INT8_MAX_ABS = 0.02


def _rss_bytes() -> int:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        ok = psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        )
        if not ok:
            raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
        return int(counters.WorkingSetSize)
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.name == "darwin":
        return int(usage)
    return int(usage) * 1024


def _save_tokenizer(tokenizer: Any, path: Path) -> None:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        backend = tokenizer._tokenizer
    path.parent.mkdir(parents=True, exist_ok=True)
    backend.save(str(path))


def _load_torch(checkpoint: Path, model_name: str):
    import torch
    from transformers import AutoModel, AutoTokenizer

    from ml.train_encoder import MultiHeadEncoder

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name, attn_implementation="eager")
    model = MultiHeadEncoder(encoder)
    model.dropout = torch.nn.Identity()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return tokenizer, model


def _export_fp32(model, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    dummy_ids = torch.ones(1, 32, dtype=torch.long)
    dummy_mask = torch.ones(1, 32, dtype=torch.long)
    names = {
        "input_names": ["input_ids", "attention_mask"],
        "output_names": ["category_logits", "secondary_logits", "urgent_logits"],
        "opset_version": 17,
    }
    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                (dummy_ids, dummy_mask),
                str(path),
                dynamo=False,
                external_data=False,
                do_constant_folding=True,
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "sequence"},
                    "attention_mask": {0: "batch", 1: "sequence"},
                    "category_logits": {0: "batch"},
                    "secondary_logits": {0: "batch"},
                    "urgent_logits": {0: "batch"},
                },
                **names,
            )
    except Exception as exc:
        print(f"legacy export failed: {type(exc).__name__}: {exc}", flush=True)
        if path.exists():
            path.unlink()
        _export_fp32_dynamo(model, dummy_ids, dummy_mask, path, names)
        return
    if not path.is_file():
        _export_fp32_dynamo(model, dummy_ids, dummy_mask, path, names)


def _export_fp32_dynamo(model, dummy_ids, dummy_mask, path: Path, names: dict) -> None:
    import torch
    from torch.export import Dim

    batch = Dim("batch", min=1, max=4)
    sequence = Dim("sequence", min=2, max=256)
    with torch.no_grad():
        program = torch.onnx.export(
            model,
            (dummy_ids, dummy_mask),
            str(path),
            dynamo=True,
            external_data=False,
            dynamic_shapes=(
                {0: batch, 1: sequence},
                {0: batch, 1: sequence},
            ),
            **names,
        )
    if program is not None and hasattr(program, "save") and not path.is_file():
        program.save(str(path))


def _reject_dropout(path: Path) -> None:
    import onnx

    proto = onnx.load(path, load_external_data=False)
    op_types = {node.op_type for node in proto.graph.node}
    if "Dropout" in op_types:
        raise RuntimeError("exported graph still contains Dropout")
    del proto


def _quantize(fp32_path: Path, int8_path: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)


def _torch_logits(model, ids: list[int], pad_id: int, pad_to: int | None):
    import torch

    sequence = list(ids)
    if pad_to is not None and len(sequence) < pad_to:
        sequence = sequence + [pad_id] * (pad_to - len(sequence))
    mask = [1] * len(ids) + [0] * (len(sequence) - len(ids))
    with torch.no_grad():
        category, secondary, urgent = model(
            torch.tensor([sequence], dtype=torch.long),
            torch.tensor([mask], dtype=torch.long),
        )
    return (
        category.float().cpu().numpy(),
        secondary.float().cpu().numpy(),
        urgent.float().cpu().numpy(),
    )


def _content_ids(tokenizer: Any, text: str) -> list[int]:
    from ml.truncation import assemble_ids

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    raw = tokenizer.encode(text, add_special_tokens=False)
    return assemble_ids(list(raw), int(cls_id), int(sep_id))


def _rows(data_dir: Path, limit: int):
    from app.text import build_input
    from ml.data import load_all

    validation = [row for row in load_all(data_dir) if row.split == "validation"]
    if not validation:
        raise RuntimeError("validation split is empty")
    rng = np.random.default_rng(42)
    count = min(limit, len(validation))
    picked = sorted(int(index) for index in rng.choice(len(validation), size=count, replace=False))
    rows = [validation[index] for index in picked]
    texts = [build_input(row.channel, row.subject, row.text) for row in rows]
    return texts


def _diff(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left - right)))


def _agree(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.argmax(left, axis=1) == np.argmax(right, axis=1)))


def _parity(
    model,
    tokenizer,
    session_fp32,
    session_int8,
    texts: list[str],
    pad_id: int,
    pad_to: int | None,
) -> dict[str, float]:
    from app.onnx_encoder import probabilities_from_logits

    torch_category = []
    torch_secondary = []
    torch_urgent = []
    fp32_category = []
    int8_category = []
    int8_secondary = []
    int8_urgent = []
    for index, text in enumerate(texts, start=1):
        ids = _content_ids(tokenizer, text)
        category, secondary, urgent = _torch_logits(model, ids, pad_id, pad_to)
        cat_p, sec_p, urg_p = probabilities_from_logits(category, secondary, urgent)
        torch_category.append(cat_p.reshape(-1, cat_p.shape[-1])[0])
        torch_secondary.append(sec_p.reshape(-1, sec_p.shape[-1])[0])
        torch_urgent.append(float(urg_p.reshape(-1)[0]))
        feeds = _feeds(ids, pad_id, pad_to)
        fp32 = session_fp32.run(["category_logits", "secondary_logits", "urgent_logits"], feeds)
        int8 = session_int8.run(["category_logits", "secondary_logits", "urgent_logits"], feeds)
        fp32_p, _, _ = probabilities_from_logits(*fp32)
        int8_p, int8_s, int8_u = probabilities_from_logits(*int8)
        fp32_category.append(fp32_p.reshape(-1, fp32_p.shape[-1])[0])
        int8_category.append(int8_p.reshape(-1, int8_p.shape[-1])[0])
        int8_secondary.append(int8_s.reshape(-1, int8_s.shape[-1])[0])
        int8_urgent.append(float(int8_u.reshape(-1)[0]))
        if index % 25 == 0 or index == len(texts):
            print(f"parity {index}/{len(texts)}", flush=True)
    torch_category_a = np.stack(torch_category)
    torch_secondary_a = np.stack(torch_secondary)
    torch_urgent_a = np.asarray(torch_urgent)
    return {
        "n": float(len(texts)),
        "fp32_category_max_abs": _diff(torch_category_a, np.stack(fp32_category)),
        "int8_category_max_abs": _diff(torch_category_a, np.stack(int8_category)),
        "int8_secondary_max_abs": _diff(torch_secondary_a, np.stack(int8_secondary)),
        "int8_urgent_max_abs": _diff(torch_urgent_a, np.asarray(int8_urgent)),
        "int8_category_argmax_agreement": _agree(torch_category_a, np.stack(int8_category)),
    }


def _feeds(ids: list[int], pad_id: int, pad_to: int | None) -> dict[str, np.ndarray]:
    sequence = list(ids)
    if pad_to is not None and len(sequence) < pad_to:
        sequence = sequence + [pad_id] * (pad_to - len(sequence))
    mask = [1] * len(ids) + [0] * (len(sequence) - len(ids))
    return {
        "input_ids": np.asarray([sequence], dtype=np.int64),
        "attention_mask": np.asarray([mask], dtype=np.int64),
    }


def _pad_gap(model, tokenizer, texts: list[str], pad_id: int) -> float:
    from app.onnx_encoder import probabilities_from_logits

    gaps = []
    for text in texts[:20]:
        ids = _content_ids(tokenizer, text)
        tight = probabilities_from_logits(*_torch_logits(model, ids, pad_id, None))[0]
        padded = probabilities_from_logits(*_torch_logits(model, ids, pad_id, 256))[0]
        gaps.append(_diff(tight, padded))
    return float(max(gaps) if gaps else 0.0)


def _session(path: Path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def _tokenizer_ids_match(tokenizer: Any, tokenizer_path: Path, texts: list[str]) -> None:
    from tokenizers import Tokenizer

    saved = Tokenizer.from_file(str(tokenizer_path))
    for text in texts:
        hf_ids = list(tokenizer.encode(text, add_special_tokens=False))
        tk_ids = list(saved.encode(text, add_special_tokens=False).ids)
        if hf_ids != tk_ids:
            raise RuntimeError("tokenizer.json ids diverged from the training tokenizer")


def _write_report(payload: dict[str, Any]) -> None:
    parity = payload["parity"]
    latency = payload["latency_validation"]
    long_latency = payload["latency_long"]
    lines = [
        "# ONNX export",
        "",
        "Graph: `intfloat/multilingual-e5-small` with the three task heads.",
        payload["checkpoint_note"],
        "",
        f"- Checkpoint: `{payload['checkpoint']}`",
        f"- int8 file: `{payload['int8_path']}` ({payload['int8_bytes']} bytes)",
        f"- fp32 file: `{payload['fp32_path']}` ({payload['fp32_bytes']} bytes)",
        f"- Padding to 256 max abs gap on 20 tickets: {payload['pad_gap_max_abs']:.6f}",
        f"- Serving pad_to: {payload['pad_to']}",
        f"- RSS after the int8 session loaded: {payload['rss_bytes'] / 1e6:.1f} MB",
        f"- Machine: {payload['cpu_count']} CPUs, {payload['processor']}",
        "",
        "| check | value |",
        "| --- | --- |",
        f"| fp32 category max abs vs PyTorch | {parity['fp32_category_max_abs']:.3e} |",
        f"| int8 category max abs vs PyTorch | {parity['int8_category_max_abs']:.6f} |",
        f"| int8 secondary max abs vs PyTorch | {parity['int8_secondary_max_abs']:.6f} |",
        f"| int8 urgent max abs vs PyTorch | {parity['int8_urgent_max_abs']:.6f} |",
        f"| int8 category argmax agreement | {parity['int8_category_argmax_agreement']:.4f} |",
        "",
        (
            "An argmax can change only when the top-two margin is within twice the "
            f"int8 category error ({2 * parity['int8_category_max_abs']:.4f}). "
            "The smoke heads are undertrained, so many tickets sit in that band. "
            "The probability match is the parity check."
        ),
        "",
        "Latency is one ticket at a time, ONNX Runtime intra-op threads = 1.",
        "This machine is not the 2 vCPU / 4 GB container.",
        "",
        "| slice | n | p50 ms | p95 ms | max ms |",
        "| --- | --- | --- | --- | --- |",
        (
            f"| validation sample | {int(latency['n'])} | {latency['p50_ms']:.1f} | "
            f"{latency['p95_ms']:.1f} | {latency['max_ms']:.1f} |"
        ),
        (
            f"| 10k-char text | {int(long_latency['n'])} | {long_latency['p50_ms']:.1f} | "
            f"{long_latency['p95_ms']:.1f} | {long_latency['max_ms']:.1f} |"
        ),
        "",
        payload["gate_note"],
        "",
    ]
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    REPORT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def export(
    checkpoint: Path,
    out_dir: Path,
    data_dir: Path,
    model_name: str,
    limit: int,
    skip_export: bool = False,
) -> dict[str, Any]:
    from app.onnx_encoder import OnnxEncoder, measure_latency
    from app.text import build_input

    print(f"loading {checkpoint}", flush=True)
    tokenizer, model = _load_torch(checkpoint, model_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = out_dir / "encoder.fp32.onnx"
    int8_path = out_dir / "encoder.int8.onnx"
    tokenizer_path = out_dir / "tokenizer.json"
    meta_path = out_dir / "encoder_meta.json"
    if skip_export:
        if not fp32_path.is_file() or not int8_path.is_file() or not tokenizer_path.is_file():
            raise SystemExit(f"missing exported files in {out_dir}")
        print("reusing exported files", flush=True)
    else:
        print("exporting fp32 onnx", flush=True)
        _export_fp32(model, fp32_path)
        _reject_dropout(fp32_path)
        print(f"fp32 bytes {fp32_path.stat().st_size}", flush=True)
        print("quantizing int8", flush=True)
        _quantize(fp32_path, int8_path)
        print(f"int8 bytes {int8_path.stat().st_size}", flush=True)
        _save_tokenizer(tokenizer, tokenizer_path)
    texts = _rows(data_dir, limit)
    _tokenizer_ids_match(tokenizer, tokenizer_path, texts[:20])
    pad_id = int(tokenizer.pad_token_id)
    pad_gap = _pad_gap(model, tokenizer, texts, pad_id)
    pad_to = 256 if pad_gap > 1e-3 else None
    print(f"pad gap {pad_gap:.6f} pad_to {pad_to}", flush=True)
    session_fp32 = _session(fp32_path)
    session_int8 = _session(int8_path)
    parity = _parity(model, tokenizer, session_fp32, session_int8, texts, pad_id, pad_to)
    print(
        "parity "
        f"fp32 {parity['fp32_category_max_abs']:.6f} "
        f"int8 {parity['int8_category_max_abs']:.6f} "
        f"argmax {parity['int8_category_argmax_agreement']:.4f}",
        flush=True,
    )
    del model, session_fp32
    gc.collect()
    smoke = "encoder_smoke" in checkpoint.as_posix()
    meta = {
        "model_name": model_name,
        "checkpoint": str(checkpoint),
        "cls_id": int(tokenizer.cls_token_id),
        "sep_id": int(tokenizer.sep_token_id),
        "pad_id": pad_id,
        "pad_to": pad_to,
        "smoke_weights": smoke,
        "opset": 17,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    encoder = OnnxEncoder.load(int8_path, tokenizer_path, meta)
    rss = _rss_bytes()
    latency = measure_latency(encoder, texts[:50], warmup=5)
    long_text = build_input("email", "subject", "word " * 2000)
    long_latency = measure_latency(encoder, [long_text] * 20, warmup=2)
    rss = max(rss, _rss_bytes())
    passed = (
        parity["fp32_category_max_abs"] <= FP32_MAX_ABS
        and parity["int8_category_max_abs"] <= INT8_MAX_ABS
        and parity["int8_secondary_max_abs"] <= INT8_MAX_ABS
        and parity["int8_urgent_max_abs"] <= INT8_MAX_ABS
    )
    if smoke:
        checkpoint_note = (
            "The checkpoint is the laptop smoke run. Its head weights are not the "
            "serving model. The matmul shapes match the final fit."
        )
        gate_note = (
            "Gate G1 was measured separately under a 2 CPU / 4 GB cap. See ml/reports/g1.md."
        )
    else:
        checkpoint_note = (
            "The checkpoint is the final fit on all 4,800 tickets for 4 epochs, "
            "the stage-A best epoch. Batch 8 with 2 accumulation steps."
        )
        gate_note = (
            f"The int8 file is {int8_path.stat().st_size / 1e6:.1f} MB, above GitHub's "
            "100 MB file limit, so it stays out of git and is copied into the image "
            "from the local artifacts directory. Parity below is against this checkpoint."
        )
    payload = {
        "checkpoint_note": checkpoint_note,
        "checkpoint": str(checkpoint),
        "fp32_path": str(fp32_path),
        "int8_path": str(int8_path),
        "tokenizer_path": str(tokenizer_path),
        "fp32_bytes": fp32_path.stat().st_size,
        "int8_bytes": int8_path.stat().st_size,
        "pad_gap_max_abs": pad_gap,
        "pad_to": pad_to,
        "rss_bytes": rss,
        "cpu_count": os.cpu_count(),
        "processor": platform.processor(),
        "parity": parity,
        "latency_validation": latency,
        "latency_long": long_latency,
        "parity_passed": passed,
        "gate_note": gate_note,
    }
    _write_report(payload)
    print(f"wrote {REPORT_MD}", flush=True)
    print(
        f"latency p50 {latency['p50_ms']:.1f} ms p95 {latency['p95_ms']:.1f} ms "
        f"long p95 {long_latency['p95_ms']:.1f} ms rss {rss / 1e6:.0f} MB",
        flush=True,
    )
    if not passed:
        raise SystemExit("onnx parity check failed; see ml/reports/onnx_latency.md")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the encoder to int8 ONNX.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--parity-tickets", type=int, default=200)
    parser.add_argument("--skip-export", action="store_true")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f"missing checkpoint {args.checkpoint}")
    export(
        args.checkpoint,
        args.out_dir,
        args.data_dir,
        args.model,
        args.parity_tickets,
        skip_export=args.skip_export,
    )


if __name__ == "__main__":
    main()
