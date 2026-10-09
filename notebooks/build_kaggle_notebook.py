"""Rebuild notebooks/kaggle_encoder.ipynb from the current training modules.

The notebook is self-contained so a Kaggle session does not need the GitHub repo.
Run this after changing encoder training code, then re-import the notebook.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "kaggle_encoder.ipynb"

VENDORED = (
    "app/__init__.py",
    "app/labels.py",
    "app/text.py",
    "app/inference.py",
    "ml/__init__.py",
    "ml/data.py",
    "ml/evaluate.py",
    "ml/truncation.py",
    "ml/train_encoder.py",
)


def _code(source: str) -> dict:
    if not source.endswith("\n"):
        source += "\n"
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source.splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
    }


def _markdown(source: str) -> dict:
    if not source.endswith("\n"):
        source += "\n"
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def main() -> None:
    cells = [
        _markdown(
            """# TensorForge encoder CV

Trains `intfloat/multilingual-e5-small` with category, secondary, and urgent heads.
Stage A (train → validation) is the number to quote. The 5-fold score is only a ranking check.

Before you run:

- Accelerator: **GPU T4**
- Internet: **On** (the notebook downloads the public encoder weights)
- Input: private dataset `tensorforge-rideeat-private`
- Do not paste an API key or upload `.env`

Use **Save Version → Save & Run All** so the job keeps running if you close the laptop.
When it finishes, download the `encoder` folder from the version output.
"""
        ),
        _code(
            """from pathlib import Path

matches = sorted(Path("/kaggle/input").rglob("train.csv"))
if len(matches) != 1:
    raise SystemExit(f"expected exactly one train.csv under /kaggle/input, found {matches}")
data_dir = matches[0].parent
print("data_dir", data_dir)
for name in ("train.csv", "validation.csv", "train.jsonl", "validation.jsonl"):
    path = data_dir / name
    print(f"{name}: {'ok' if path.is_file() else 'MISSING'} {path.stat().st_size if path.is_file() else 0}")
"""
        ),
        _code(
            """from pathlib import Path

Path("app").mkdir(exist_ok=True)
Path("ml").mkdir(exist_ok=True)
print("package dirs ready")
"""
        ),
    ]
    for relative in VENDORED:
        text = (ROOT / relative).read_text(encoding="utf-8")
        if not text.endswith("\n"):
            text += "\n"
        cells.append(_code(f"%%writefile {relative}\n{text}"))
    cells.append(
        _code(
            """import os
import subprocess
import sys

env = os.environ.copy()
env["PYTHONPATH"] = "/kaggle/working"
env["TOKENIZERS_PARALLELISM"] = "false"
env["HF_HUB_DISABLE_TELEMETRY"] = "1"
command = [
    sys.executable,
    "-m",
    "ml.train_encoder",
    "--data-dir",
    str(data_dir),
    "--out-dir",
    "/kaggle/working/encoder",
    "--epochs",
    "6",
    "--batch-size",
    "16",
    "--patience",
    "2",
]
print(" ".join(command), flush=True)
subprocess.check_call(command, cwd="/kaggle/working", env=env)
print("finished", flush=True)
print("download /kaggle/working/encoder (summary.json, stage_a/, fold_0..4/)", flush=True)
"""
        )
    )
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }
    OUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
