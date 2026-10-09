from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.inference import DEFAULT_ARTIFACT, Engine
from ml.data import DATA_DIR, load_all

REPRO_PATH = Path("ml/reports/repro_sample.json")


@pytest.mark.skipif(
    not DEFAULT_ARTIFACT.is_file() or not REPRO_PATH.is_file(),
    reason="baseline artifact not trained",
)
@pytest.mark.skipif(
    not all((DATA_DIR / name).is_file() for name in ("train.csv", "validation.csv")),
    reason="organizer dataset not downloaded",
)
def test_artifact_reproduces_saved_predictions():
    sample = json.loads(REPRO_PATH.read_text(encoding="utf-8"))
    rows = {row.ticket_id: row for row in load_all()}
    engine = Engine.load(DEFAULT_ARTIFACT)
    assert engine.model_version
    for expected in sample:
        row = rows[expected["ticket_id"]]
        first = engine.predict(
            {
                "channel": row.channel,
                "subject": row.subject,
                "text": row.text,
                "ticket_id": row.ticket_id,
            }
        )
        second = engine.predict(
            {
                "channel": row.channel,
                "subject": row.subject,
                "text": row.text,
                "ticket_id": row.ticket_id,
            }
        )
        assert first == second
        for key in (
            "category",
            "secondary_category",
            "is_urgent",
            "confidence",
            "team",
            "model_version",
        ):
            assert first[key] == expected[key]
