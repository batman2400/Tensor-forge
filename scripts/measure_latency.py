"""Measure TensorForge latency from wherever this script runs.

    python scripts/measure_latency.py --base-url https://host --label vm-public --out result.json

Run it once from a client close to the server and once from a remote client: the gap is the
network, not the application. Reports single-prediction, 100-ticket batch, job-submission and
job-polling latency. The key comes from the API_KEY environment variable or the local `.env`
and is never printed or written to the output file.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]

SAMPLES = [
    ("chat", "", "I was charged twice for my ride, please refund the extra payment."),
    ("email", "Missing item", "My order arrived without the fries and the drink I paid for."),
    ("call_transcript", "", "The driver was rude and drove dangerously, I felt unsafe in the car."),
    ("chat", "", "The app crashes every time I open the promo code screen."),
    (
        "email",
        "Lost phone",
        "I left my phone on the back seat of the car after my trip last night.",
    ),
]


def load_key() -> str:
    key = os.environ.get("API_KEY", "")
    env_file = ROOT / ".env"
    if not key and env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("API_KEY="):
                key = line.split("=", 1)[1].strip().strip("'\"")
    if not key:
        sys.exit("API_KEY not set (env or .env)")
    return key


def make_tickets(count: int, prefix: str) -> list[dict[str, str]]:
    return [
        {
            "ticket_id": f"{prefix}{i}",
            "channel": SAMPLES[i % len(SAMPLES)][0],
            "subject": SAMPLES[i % len(SAMPLES)][1],
            "text": f"{SAMPLES[i % len(SAMPLES)][2]} Reference {i}.",
        }
        for i in range(count)
    ]


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    return ordered[min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))]


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "n": len(values),
        "p50": round(pct(values, 0.5), 4),
        "p95": round(pct(values, 0.95), 4),
        "max": round(max(values), 4) if values else float("nan"),
        "mean": round(statistics.fmean(values), 4) if values else float("nan"),
    }


def run_job(
    client: httpx.Client, size: int, headers: dict[str, str], deadline: float
) -> dict[str, Any]:
    tickets = make_tickets(size, f"m{size}-")
    body = json.dumps({"tickets": tickets}).encode()
    started = time.perf_counter()
    response = client.post(
        "/batch/jobs",
        content=body,
        headers={**headers, "Content-Type": "application/json"},
        timeout=120,
    )
    submit_s = time.perf_counter() - started
    out: dict[str, Any] = {
        "tickets": size,
        "body_bytes": len(body),
        "submit_status": response.status_code,
        "submit_seconds": round(submit_s, 3),
    }
    if response.status_code != 202:
        out["error"] = response.text[:120]
        return out
    job_id = response.json()["job_id"]
    polls: list[float] = []
    health: list[float] = []
    status = "queued"
    # Dedicated keep-alive client for polling, as a harness would use.
    with httpx.Client(base_url=str(client.base_url), headers=headers, timeout=30) as poller:
        while time.perf_counter() - started < deadline:
            t = time.perf_counter()
            polled = poller.get(f"/batch/jobs/{job_id}")
            polls.append(time.perf_counter() - t)
            t = time.perf_counter()
            poller.get("/health")
            health.append(time.perf_counter() - t)
            status = polled.json().get("status", "?")
            if status not in ("queued", "running"):
                break
            time.sleep(1.0)
        t = time.perf_counter()
        results = poller.get(f"/batch/jobs/{job_id}/results", params={"limit": 100})
        out["results_first_page_seconds"] = round(time.perf_counter() - t, 3)
        out["results_status"] = results.status_code
        poller.delete(f"/batch/jobs/{job_id}")
    out["final_status"] = status
    out["completed_seconds"] = round(time.perf_counter() - started, 1)
    out["poll"] = summarize(polls)
    out["health_during_job"] = summarize(health)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("TF_BASE_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--label", default="client")
    ap.add_argument("--singles", type=int, default=100)
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--jobs", type=int, nargs="*", default=[2000, 5000])
    ap.add_argument("--deadline", type=float, default=900.0, help="per-job completion budget (s)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    key = load_key()
    headers = {"X-API-Key": key}
    result: dict[str, Any] = {
        "label": args.label,
        "base_url": args.base_url,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with httpx.Client(base_url=args.base_url.rstrip("/"), headers=headers, timeout=60) as client:
        health = client.get("/health")
        result["model_version"] = health.json().get("model_version")
        # Connection setup cost, measured separately from steady-state requests.
        setup: list[float] = []
        for _ in range(5):
            with httpx.Client(base_url=args.base_url.rstrip("/"), timeout=30) as fresh:
                t = time.perf_counter()
                fresh.get("/health")
                setup.append(time.perf_counter() - t)
        result["new_connection_health"] = summarize(setup)

        warm = {"channel": "chat", "text": SAMPLES[0][2]}
        for _ in range(3):
            client.post("/predict", json=warm)
        singles: list[float] = []
        failures = 0
        for i in range(args.singles):
            channel, subject, text = SAMPLES[i % len(SAMPLES)]
            t = time.perf_counter()
            r = client.post(
                "/predict",
                json={"ticket_id": f"s{i}", "channel": channel, "subject": subject, "text": text},
            )
            singles.append(time.perf_counter() - t)
            failures += r.status_code != 200
        result["single_predict"] = {**summarize(singles), "non_200": failures}

        batch_times: list[float] = []
        batch = {"tickets": make_tickets(100, "b")}
        for _ in range(args.batches):
            t = time.perf_counter()
            r = client.post("/predict/batch", json=batch)
            batch_times.append(time.perf_counter() - t)
            failures += r.status_code != 200
        result["batch_100"] = summarize(batch_times)

        result["jobs"] = [run_job(client, n, headers, args.deadline) for n in args.jobs]

    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
