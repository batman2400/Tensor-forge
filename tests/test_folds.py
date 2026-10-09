from __future__ import annotations

import json
from collections import Counter, defaultdict

import pytest

from ml.data import FOLDS_PATH, N_SPLITS, assign_folds, load_all, load_folds


def test_folds_file_shape():
    payload = json.loads(FOLDS_PATH.read_text(encoding="utf-8"))
    assert payload["seed"] == 42
    assert payload["n_splits"] == N_SPLITS
    assert payload["stratify"] == "category|language"
    fold_of = payload["fold_of"]
    assert len(fold_of) == 4800
    assert set(fold_of) == set(fold_of.keys())
    assert set(fold_of.values()) == set(range(N_SPLITS))
    assert list(fold_of) == sorted(fold_of)


@pytest.mark.skipif(
    not (FOLDS_PATH.parents[1] / "data" / "train.csv").is_file(), reason="dataset not downloaded"
)
def test_folds_match_generator_and_balance_each_cell():
    rows = load_all()
    saved = load_folds()
    assert saved == assign_folds(rows)
    assert set(saved) == {row.ticket_id for row in rows}

    groups = defaultdict(list)
    for row in rows:
        groups[(row.category, row.language)].append(saved[row.ticket_id])
    for fold_ids in groups.values():
        counts = Counter(fold_ids)
        spread = [counts.get(fold, 0) for fold in range(N_SPLITS)]
        assert max(spread) - min(spread) <= 1

    sizes = [sum(1 for fold in saved.values() if fold == index) for index in range(N_SPLITS)]
    # Each cell differs by at most one ticket per fold, so the global gap is at most one per cell.
    assert max(sizes) - min(sizes) <= 66
