from __future__ import annotations

import re

import yaml

from app.labels import CATEGORIES, CHANNELS, TEAM_BY_CATEGORY
from tests.helpers import ROOT


def _team_table(description: str) -> dict[str, str]:
    table: dict[str, str] = {}
    for line in description.splitlines():
        match = re.match(r"^\s*\|\s*([a-z_]+)\s*\|\s*(.+?)\s*\|\s*$", line)
        if match and match.group(1) != "category":
            table[match.group(1)] = match.group(2)
    return table


def test_labels_match_openapi_enums_and_team_table():
    spec = yaml.safe_load(
        (ROOT / "spec" / "tensorforge-phase2-openapi-v2.yaml").read_text(encoding="utf-8")
    )
    schemas = spec["components"]["schemas"]
    assert CATEGORIES == tuple(schemas["Category"]["enum"])
    assert CHANNELS == tuple(schemas["Channel"]["enum"])
    assert [TEAM_BY_CATEGORY[category] for category in CATEGORIES] == schemas["Team"]["enum"]
    assert _team_table(spec["info"]["description"]) == TEAM_BY_CATEGORY
