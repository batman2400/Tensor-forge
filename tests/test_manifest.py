"""Manifest version is the hash prefix, and a changed file is rejected."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.manifest import file_sha256, stamp, verify


def test_version_is_the_manifest_hash_prefix(tmp_path: Path):
    artifact = tmp_path / "classical.joblib"
    artifact.write_bytes(b"weights")
    stamped = stamp(
        {
            "encoder_enabled": False,
            "files": {"classical.joblib": file_sha256(artifact)},
            "serving": "classical",
        }
    )
    assert stamped["model_version"] == "v1.0.0-" + stamped["manifest_sha256"][:8]
    verify(stamped, tmp_path)


def test_changed_file_fails_verification(tmp_path: Path):
    artifact = tmp_path / "classical.joblib"
    artifact.write_bytes(b"weights")
    stamped = stamp({"files": {"classical.joblib": file_sha256(artifact)}})
    artifact.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        verify(stamped, tmp_path)


def test_unsafe_path_is_rejected(tmp_path: Path):
    stamped = stamp({"files": {"../secret": "abc"}})
    with pytest.raises(RuntimeError, match="unsafe manifest path"):
        verify(stamped, tmp_path)


def test_service_load_uses_the_frozen_version():
    from app.inference import DEFAULT_ARTIFACT, MANIFEST_PATH, Engine

    if not DEFAULT_ARTIFACT.is_file() or not MANIFEST_PATH.is_file():
        pytest.skip("manifest not frozen")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    frozen = Engine.load()
    plain = Engine.load(DEFAULT_ARTIFACT)
    ticket = {
        "channel": "chat",
        "subject": "refund",
        "text": "my payment was charged twice",
        "ticket_id": "t",
    }
    left = frozen.predict(ticket)
    assert frozen.model_version.startswith("v1.0.0-")
    assert left["model_version"] == frozen.model_version
    if manifest.get("encoder_enabled"):
        assert frozen.encoder is not None
    else:
        right = plain.predict(ticket)
        assert frozen.encoder is None
        for key in ("category", "secondary_category", "is_urgent", "team", "confidence"):
            assert left[key] == right[key]


def test_round_trip_through_pretty_json(tmp_path: Path):
    artifact = tmp_path / "classical.joblib"
    artifact.write_bytes(b"weights")
    stamped = stamp({"files": {"classical.joblib": file_sha256(artifact)}})
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(stamped, indent=2) + "\n", encoding="utf-8")
    verify(json.loads(path.read_text(encoding="utf-8")), tmp_path)
