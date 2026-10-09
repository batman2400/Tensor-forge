"""Dataset loading and the fixed 5-fold split.

CSV is read with the csv module (text fields contain embedded newlines).
`folds.json` stores ticket ids only.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FOLDS_PATH = ROOT / "ml" / "folds.json"

FOLD_SEED = 42
N_SPLITS = 5


@dataclass(frozen=True, slots=True)
class TicketRow:
    ticket_id: str
    channel: str
    subject: str
    text: str
    language: str
    category: str
    secondary_category: str | None
    is_urgent: bool
    split: str


def _load_csv(path: Path, split: str) -> list[TicketRow]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Download the dataset into data/ (see data/README.md)."
        )
    rows: list[TicketRow] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            secondary = raw["secondary_category"] or None
            rows.append(
                TicketRow(
                    ticket_id=raw["ticket_id"],
                    channel=raw["channel"],
                    subject=raw["subject"],
                    text=raw["text"],
                    language=raw["language"],
                    category=raw["category"],
                    secondary_category=secondary,
                    is_urgent=raw["is_urgent"] == "true",
                    split=split,
                )
            )
    return rows


def load_split(split: str, data_dir: Path | None = None) -> list[TicketRow]:
    if split not in {"train", "validation"}:
        raise ValueError(f"unknown split {split!r}")
    directory = DATA_DIR if data_dir is None else data_dir
    return _load_csv(directory / f"{split}.csv", split)


def load_all(data_dir: Path | None = None) -> list[TicketRow]:
    """Train rows first, then validation. Order is stable."""
    return load_split("train", data_dir) + load_split("validation", data_dir)


def assign_folds(
    rows: list[TicketRow],
    n_splits: int = N_SPLITS,
    seed: int = FOLD_SEED,
) -> dict[str, int]:
    """Round-robin inside each category x language cell.

    Two cells have fewer than five tickets, so StratifiedKFold(n_splits=5) cannot
    be used on the joint key. Shuffling inside the cell and dealing round-robin
    still balances every cell across folds (counts differ by at most one).
    """
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in rows:
        groups[(row.category, row.language)].append(row.ticket_id)

    rng = np.random.default_rng(seed)
    fold_of: dict[str, int] = {}
    for key in sorted(groups):
        ids = groups[key][:]
        rng.shuffle(ids)
        for offset, ticket_id in enumerate(ids):
            fold_of[ticket_id] = offset % n_splits
    return fold_of


def write_folds(
    rows: list[TicketRow] | None = None,
    path: Path = FOLDS_PATH,
    n_splits: int = N_SPLITS,
    seed: int = FOLD_SEED,
) -> dict[str, int]:
    rows = load_all() if rows is None else rows
    fold_of = assign_folds(rows, n_splits=n_splits, seed=seed)
    payload = {
        "seed": seed,
        "n_splits": n_splits,
        "stratify": "category|language",
        "method": "round-robin within each category-language cell after a seeded shuffle",
        "fold_of": {ticket_id: fold_of[ticket_id] for ticket_id in sorted(fold_of)},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return fold_of


def load_folds(path: Path = FOLDS_PATH) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {ticket_id: int(fold) for ticket_id, fold in payload["fold_of"].items()}


def main() -> None:
    rows = load_all()
    fold_of = write_folds(rows)
    sizes = [sum(1 for fold in fold_of.values() if fold == i) for i in range(N_SPLITS)]
    print(f"wrote {FOLDS_PATH} ({len(fold_of)} ids, fold sizes {sizes})")


if __name__ == "__main__":
    main()
