"""Time /health and job polling while a hosted job runs.

    python scripts/probe_poll.py --base-url https://host --jobs 2000

Reads API_KEY from the environment and never prints it.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


def _percentile(samples: list[float], p: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    rank = p * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _tickets(count: int) -> list[dict[str, str]]:
    root = Path(__file__).resolve().parents[1]
    rows: list[dict[str, str]] = []
    if (root / "data" / "validation.csv").is_file():
        from ml.data import load_split

        for row in load_split("validation", root / "data"):
            rows.append({"channel": row.channel, "subject": row.subject, "text": row.text})
    if not rows:
        rows = [{"channel": "chat", "subject": "refund", "text": "my payment was charged twice"}]
    tickets = []
    for index in range(count):
        source = rows[index % len(rows)]
        tickets.append(
            {
                "ticket_id": f"poll-{index:05d}",
                "channel": source["channel"],
                "subject": source["subject"],
                "text": source["text"],
            }
        )
    return tickets


def _call(
    url: str, key: str, method: str, body: bytes | None, timeout: float
) -> tuple[int, bytes, float]:
    headers = {"X-API-Key": key}
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)  # noqa: S310
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return response.status, response.read(), (time.perf_counter() - started) * 1000
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), (time.perf_counter() - started) * 1000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--jobs", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=5)
    args = parser.parse_args()
    key = os.environ.get("API_KEY")
    if not key:
        raise SystemExit("API_KEY is unset")
    base = args.base_url.rstrip("/")
    body = json.dumps({"tickets": _tickets(args.jobs)}).encode()
    status, payload, submit_ms = _call(base + "/batch/jobs", key, "POST", body, 30)
    if status != 202:
        raise SystemExit(f"submit {status} {payload[:180]!r}")
    job_id = json.loads(payload)["job_id"]
    health_ms: list[float] = []
    poll_ms: list[float] = []
    failures = 0
    deadline = time.perf_counter() + 35 * 60
    last = {}
    while time.perf_counter() < deadline:
        try:
            health_status, _, elapsed = _call(base + "/health", "", "GET", None, args.timeout)
            # Health is public. Drop the key header by calling without auth below.
        except (TimeoutError, urllib.error.URLError):
            failures += 1
            health_status = 0
            elapsed = args.timeout * 1000
        if health_status == 200:
            health_ms.append(elapsed)
        elif health_status != 0:
            failures += 1
        try:
            poll_status, poll_body, elapsed = _call(
                f"{base}/batch/jobs/{job_id}", key, "GET", None, args.timeout
            )
        except (TimeoutError, urllib.error.URLError):
            failures += 1
            time.sleep(0.5)
            continue
        if poll_status != 200:
            failures += 1
            time.sleep(0.5)
            continue
        poll_ms.append(elapsed)
        last = json.loads(poll_body)
        if last.get("status") in {"succeeded", "failed", "cancelled"}:
            break
        time.sleep(0.4)
    print(
        json.dumps(
            {
                "status": last.get("status"),
                "processed": last.get("processed"),
                "total": last.get("total"),
                "submit_ms": round(submit_ms, 1),
                "failures": failures,
                "health_n": len(health_ms),
                "health_p50_ms": round(_percentile(health_ms, 0.5), 1),
                "health_p95_ms": round(_percentile(health_ms, 0.95), 1),
                "health_max_ms": round(max(health_ms) if health_ms else 0, 1),
                "poll_n": len(poll_ms),
                "poll_p50_ms": round(_percentile(poll_ms, 0.5), 1),
                "poll_p95_ms": round(_percentile(poll_ms, 0.95), 1),
                "poll_max_ms": round(max(poll_ms) if poll_ms else 0, 1),
            }
        ),
        flush=True,
    )
    if last.get("status") != "succeeded" or failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
