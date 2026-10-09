"""Write artifacts/manifest.json for the model the service loads.

Pass ``--encoder`` only after the int8 graph in ``artifacts/`` is the final
fit, not the laptop smoke run.

    python -m ml.freeze
    python -m ml.freeze --encoder
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from app.manifest import file_sha256, stamp

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
FUSION_REPORT = ROOT / "ml" / "reports" / "fusion.json"
ENCODER_FILES = ("encoder.int8.onnx", "tokenizer.json", "encoder_meta.json")


def _git_commit() -> str:
    """HEAD commit without spawning git. Packed refs are enough for a normal clone."""
    git_dir = ROOT / ".git"
    head_path = git_dir / "HEAD"
    if not head_path.is_file():
        return "unknown"
    head = head_path.read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head
    ref_path = git_dir / head.removeprefix("ref: ")
    if ref_path.is_file():
        return ref_path.read_text(encoding="utf-8").strip()
    packed = git_dir / "packed-refs"
    if not packed.is_file():
        return "unknown"
    target = head.removeprefix("ref: ")
    for line in packed.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or line.startswith("^"):
            continue
        sha, _, name = line.partition(" ")
        if name == target:
            return sha
    return "unknown"


def _metrics() -> dict[str, float]:
    if not FUSION_REPORT.is_file():
        return {}
    payload = json.loads(FUSION_REPORT.read_text(encoding="utf-8"))
    stage_a = payload.get("stage_a") or {}
    classical = payload.get("classical_stage_a") or {}
    metrics: dict[str, float] = {}
    if "macro_f1" in stage_a:
        metrics["fusion_stage_a_macro_f1"] = float(stage_a["macro_f1"])
    if "macro_f1" in classical:
        metrics["classical_stage_a_macro_f1"] = float(classical["macro_f1"])
    return metrics


def build(encoder_enabled: bool = False) -> dict:
    classical = ARTIFACTS / "classical.joblib"
    if not classical.is_file():
        raise SystemExit(f"missing {classical}")
    fusion_path = ARTIFACTS / "fusion.json"
    if FUSION_REPORT.is_file():
        payload = json.loads(FUSION_REPORT.read_text(encoding="utf-8"))
        fusion_path.write_text(
            json.dumps({"settings": payload["settings"]}, indent=2) + "\n",
            encoding="utf-8",
        )
    files = {"classical.joblib": file_sha256(classical)}
    if fusion_path.is_file():
        files["fusion.json"] = file_sha256(fusion_path)
    if encoder_enabled:
        for name in ENCODER_FILES:
            path = ARTIFACTS / name
            if not path.is_file():
                raise SystemExit(f"missing {path}")
            files[name] = file_sha256(path)
    if encoder_enabled:
        license_notes = (
            "Classical TF-IDF models were trained for this project. "
            "The encoder is intfloat/multilingual-e5-small (MIT), exported to int8 ONNX."
        )
    else:
        license_notes = (
            "Classical TF-IDF models were trained for this project. "
            "intfloat/multilingual-e5-small is MIT; it is not in this manifest "
            "while encoder_enabled is false."
        )
    body = {
        "encoder_enabled": encoder_enabled,
        "files": files,
        "training_date": date.today().isoformat(),
        "git_commit": _git_commit(),
        "metrics": _metrics(),
        "license_notes": license_notes,
        "serving": "classical" if not encoder_enabled else "fusion",
    }
    return stamp(body)


def write(encoder_enabled: bool = False) -> Path:
    manifest = build(encoder_enabled=encoder_enabled)
    path = ARTIFACTS / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Write artifacts/manifest.json.")
    parser.add_argument("--encoder", action="store_true")
    args = parser.parse_args()
    path = write(encoder_enabled=args.encoder)
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(f"wrote {path} version {payload['model_version']}", flush=True)


if __name__ == "__main__":
    main()
