# data/

The organizer dataset is **not redistributed** in this repository. Everything in this folder except
this file and `.gitkeep` is gitignored (and blocked by the pre-commit hook).

## Get the data

Download the official "TensorForge 2.0 - MVP Resources DataSet" folder from the organizers' link:

<https://drive.google.com/drive/folders/16S4yfPFzjbTQUb8uD8g0GPFCyK1LPWYm?usp=sharing>

Place these files directly in `data/`:

| File | Rows | Notes |
| --- | --- | --- |
| `train.csv` / `train.jsonl` | 4,000 | identical content, two formats |
| `validation.csv` / `validation.jsonl` | 800 | identical content, two formats |
| `DATA_NOTES.md` | - | organizer notes on labels and rules |

Columns: `ticket_id, channel, subject, text, language, category, secondary_category, is_urgent`.
`language` is analysis-only and is **not** available at inference time.

## Check it

```powershell
python scripts/verify_assets.py
```

CSV text fields contain embedded newlines, so always read with a real CSV parser
(`csv` module or `pandas.read_csv(..., keep_default_na=False)`), never line by line.
