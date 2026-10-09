#!/usr/bin/env python3
"""Block commits that would leak the TensorForge API key or other secrets.

Used by .githooks/pre-commit (staged content), .githooks/commit-msg (message)
and optionally CI (`--all`). Standard library only, so it runs with any Python.

What it blocks:
  1. Any staged file that should never be committed (.env, private keys,
     organizer dataset files).
  2. Anything that looks like a TensorForge key: the `tf2_` prefix followed by
     16+ hex characters.
  3. The exact value of API_KEY from the local .env file or the environment,
     even if its format ever differs from the pattern above.

Secrets are never printed; matches are reported as file:line with a redacted
preview.

Usage:
  python scripts/check_secrets.py --staged
  python scripts/check_secrets.py --all
  python scripts/check_secrets.py --message .git/COMMIT_EDITMSG
  python scripts/check_secrets.py --paths README.md app/main.py
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import subprocess
import sys
from pathlib import Path

KEY_PATTERN = re.compile(rb"tf2_[0-9a-fA-F]{16,}")
MIN_EXACT_SECRET_LEN = 12

# Paths that must never be committed (matched against the repo-relative POSIX path).
FORBIDDEN_PATH_GLOBS = [
    ".env",
    ".env.*",
    "**/.env",
    "**/.env.*",
    "*.pem",
    "*.key",
    "secrets/*",
    "data/*.csv",
    "data/*.jsonl",
    "data/*.parquet",
]
ALLOWED_PATHS = {".env.example"}

# This file defines the patterns above and must not flag itself.
SELF = "scripts/check_secrets.py"


def repo_root() -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(out.stdout.strip())


def git_lines(args: list[str], root: Path) -> list[str]:
    out = subprocess.run(["git", *args], capture_output=True, cwd=root, check=True).stdout
    return [p.decode("utf-8", "surrogateescape") for p in out.split(b"\0") if p]


def local_exact_secrets(root: Path) -> list[bytes]:
    """Exact secret values from the local .env and environment (never printed)."""
    values: set[str] = set()
    env_file = root / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            if name.strip().upper().endswith(("KEY", "TOKEN", "SECRET", "PASSWORD")):
                values.add(value.strip().strip("'\""))
    if os.environ.get("API_KEY"):
        values.add(os.environ["API_KEY"])
    return [v.encode() for v in values if len(v) >= MIN_EXACT_SECRET_LEN]


def forbidden_path(rel: str) -> bool:
    if rel in ALLOWED_PATHS:
        return False
    return any(fnmatch.fnmatch(rel, g) for g in FORBIDDEN_PATH_GLOBS)


def scan_bytes(rel: str, data: bytes, exact: list[bytes], problems: list[str]) -> None:
    if rel == SELF:
        return
    for lineno, line in enumerate(data.split(b"\n"), start=1):
        if KEY_PATTERN.search(line):
            problems.append(f"{rel}:{lineno}: looks like a TensorForge API key (tf2_...)")
            continue
        for secret in exact:
            if secret in line:
                problems.append(f"{rel}:{lineno}: contains the exact value of a local secret")
                break


def check_paths(root: Path, rels: list[str], read_staged: bool) -> list[str]:
    problems: list[str] = []
    exact = local_exact_secrets(root)
    for rel in rels:
        posix = rel.replace("\\", "/")
        if forbidden_path(posix):
            problems.append(f"{posix}: this file must never be committed (see .gitignore)")
            continue
        if read_staged:
            res = subprocess.run(["git", "show", f":{rel}"], capture_output=True, cwd=root)
            if res.returncode != 0:
                continue
            data = res.stdout
        else:
            path = root / rel
            if not path.is_file():
                continue
            data = path.read_bytes()
        scan_bytes(posix, data, exact, problems)
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--staged", action="store_true", help="scan staged changes")
    mode.add_argument("--all", action="store_true", help="scan all tracked files")
    mode.add_argument("--message", metavar="FILE", help="scan a commit message file")
    mode.add_argument("--paths", nargs="+", metavar="PATH", help="scan specific files")
    args = ap.parse_args()

    root = repo_root()

    if args.message:
        data = Path(args.message).read_bytes()
        problems: list[str] = []
        scan_bytes("<commit message>", data, local_exact_secrets(root), problems)
    elif args.staged:
        rels = git_lines(["diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR"], root)
        problems = check_paths(root, rels, read_staged=True)
    elif args.all:
        rels = git_lines(["ls-files", "-z"], root)
        problems = check_paths(root, rels, read_staged=False)
    else:
        problems = check_paths(root, args.paths, read_staged=False)

    if problems:
        print("\nSECRET CHECK FAILED - commit blocked:\n", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        print(
            "\nRemove the secret (keep it only in .env), unstage the file, and retry.\n"
            "If a real key was exposed, tell the organizers; do not rotate it yourself.\n",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
