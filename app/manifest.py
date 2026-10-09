"""Artifact manifest: file hashes and the model_version string.

`model_version` is `v1.0.0-` plus the first 8 hex characters of the SHA-256 of
the manifest body. That body is every field except `model_version` and
`manifest_sha256`, so the version can be checked without hashing itself.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

VERSION_PREFIX = "v1.0.0-"
_EXCLUDED = frozenset({"model_version", "manifest_sha256"})


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(manifest: dict[str, Any]) -> bytes:
    body = {key: manifest[key] for key in manifest if key not in _EXCLUDED}
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


def manifest_sha256(manifest: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(manifest)).hexdigest()


def version_for(digest: str) -> str:
    return f"{VERSION_PREFIX}{digest[:8]}"


def stamp(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with `manifest_sha256` and `model_version` filled in."""
    stamped = {key: value for key, value in manifest.items() if key not in _EXCLUDED}
    digest = manifest_sha256(stamped)
    stamped["manifest_sha256"] = digest
    stamped["model_version"] = version_for(digest)
    return stamped


def verify(manifest: dict[str, Any], root: Path) -> None:
    """Raise RuntimeError if the version or any listed file hash does not match."""
    digest = manifest_sha256(manifest)
    if manifest.get("manifest_sha256") != digest:
        raise RuntimeError("manifest hash mismatch")
    if manifest.get("model_version") != version_for(digest):
        raise RuntimeError("model_version does not match the manifest hash")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError("manifest files are missing")
    for name, expected in files.items():
        relative = Path(str(name))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"unsafe manifest path {name}")
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"missing artifact {name}")
        actual = file_sha256(path)
        if actual != expected:
            raise RuntimeError(f"hash mismatch for {name}")
