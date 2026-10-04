#!/usr/bin/env python3
"""Verify that the organizer assets load correctly and map onto each other.

Checks:
  1. OpenAPI YAML parses and exposes every required operation.
  2. Every JSON Schema (and the bundle) is a valid draft 2020-12 schema.
  3. Category / Team / Channel enums agree across the YAML, each schema file and the bundle,
     and the category -> team table in the YAML description matches the schema's if/then rules.
  4. Every example in the OpenAPI YAML validates against the matching JSON Schema, and the
     consistency rules actually reject bad responses (negative controls).
  5. The dataset (CSV and JSONL) loads with the documented columns and row counts, CSV and
     JSONL are identical, labels are inside the enums, and the documented label rules hold.
  6. Every dataset row maps onto a valid PredictRequest, and its gold labels map onto a valid
     PredictResponse (team derived from category) under the JSON Schemas.

Usage:
  python scripts/verify_assets.py            # everything
  python scripts/verify_assets.py --no-data  # spec + schemas only (dataset not downloaded)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parent.parent
SPEC_DIR = ROOT / "spec"
DATA_DIR = ROOT / "data"
YAML_PATH = SPEC_DIR / "tensorforge-phase2-openapi-v2.yaml"
BUNDLE_PATH = SPEC_DIR / "tensorforge-schemas.json"

EXPECTED_OPERATIONS = {
    ("get", "/health"),
    ("post", "/predict"),
    ("post", "/predict/batch"),
    ("post", "/batch/jobs"),
    ("get", "/batch/jobs/{job_id}"),
    ("delete", "/batch/jobs/{job_id}"),
    ("get", "/batch/jobs/{job_id}/results"),
}
SCHEMA_FILES = [
    "predict_request",
    "predict_response",
    "batch_request",
    "batch_response",
    "batch_job_request",
    "batch_job_status",
    "batch_job_results",
    "health_response",
    "error_response",
]
DATA_COLUMNS = [
    "ticket_id",
    "channel",
    "subject",
    "text",
    "language",
    "category",
    "secondary_category",
    "is_urgent",
]
EXPECTED_ROWS = {"train": 4000, "validation": 800}
LANGUAGES = {"en", "si", "ta", "singlish", "tanglish", "mixed"}


class Report:
    def __init__(self) -> None:
        self.failures = 0
        self.warnings = 0

    def ok(self, msg: str) -> None:
        print(f"  [ OK ] {msg}")

    def fail(self, msg: str) -> None:
        self.failures += 1
        print(f"  [FAIL] {msg}")

    def warn(self, msg: str) -> None:
        self.warnings += 1
        print(f"  [WARN] {msg}")

    def check(self, cond: bool, ok_msg: str, fail_msg: str) -> bool:
        (self.ok if cond else self.fail)(ok_msg if cond else fail_msg)
        return cond


def section(title: str) -> None:
    print(f"\n== {title}")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- spec


def parse_team_table(description: str) -> dict[str, str]:
    """Parse the `| category | team |` markdown table in the OpenAPI description."""
    table: dict[str, str] = {}
    for line in description.splitlines():
        m = re.match(r"^\s*\|\s*([a-z_]+)\s*\|\s*(.+?)\s*\|\s*$", line)
        if m and m.group(1) not in {"category"}:
            table[m.group(1)] = m.group(2)
    return table


def schema_team_rules(defs: dict) -> dict[str, str]:
    """Extract category -> team from the if/then rules in PredictResponse."""
    rules: dict[str, str] = {}
    for rule in defs["PredictResponse"].get("allOf", []):
        cat = rule["if"]["properties"]["category"]["const"]
        team = rule["then"]["properties"].get("team", {}).get("const")
        if team:
            rules[cat] = team
    return rules


def verify_spec(rep: Report):
    section("OpenAPI YAML")
    spec = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))
    rep.check(spec.get("openapi", "").startswith("3.0"), f"openapi {spec.get('openapi')}", "unexpected openapi version")
    rep.ok(f"title: {spec['info']['title']} (v{spec['info']['version']})")

    ops = {
        (method, path)
        for path, item in spec["paths"].items()
        for method in item
        if method in {"get", "post", "put", "patch", "delete"}
    }
    missing = EXPECTED_OPERATIONS - ops
    extra = ops - EXPECTED_OPERATIONS
    rep.check(not missing, f"all {len(EXPECTED_OPERATIONS)} required operations present", f"missing operations: {sorted(missing)}")
    if extra:
        rep.warn(f"additional operations in spec: {sorted(extra)}")

    comps = spec["components"]["schemas"]
    yaml_categories = comps["Category"]["enum"]
    yaml_teams = comps["Team"]["enum"]
    yaml_channels = comps["Channel"]["enum"]
    rep.ok(f"{len(yaml_categories)} categories, {len(yaml_teams)} teams, channels={yaml_channels}")

    table = parse_team_table(spec["info"]["description"])
    rep.check(
        set(table) == set(yaml_categories),
        f"category->team table in YAML description covers all {len(table)} categories",
        f"team table categories differ from enum: {set(table) ^ set(yaml_categories)}",
    )
    rep.check(
        set(table.values()) == set(yaml_teams),
        "team table values equal the Team enum",
        f"team table values differ from Team enum: {set(table.values()) ^ set(yaml_teams)}",
    )
    return spec, table, yaml_categories, yaml_teams, yaml_channels


# ------------------------------------------------------------------------ schemas


def verify_schemas(rep: Report, spec: dict, table, yaml_categories, yaml_teams, yaml_channels):
    section("JSON Schemas (draft 2020-12)")
    schemas: dict[str, dict] = {}
    for name in SCHEMA_FILES:
        path = SPEC_DIR / f"{name}.schema.json"
        if not path.is_file():
            rep.fail(f"missing {path.name}")
            continue
        schema = load_json(path)
        try:
            Draft202012Validator.check_schema(schema)
            schemas[name] = schema
            rep.ok(f"{path.name} is a valid schema")
        except Exception as exc:  # noqa: BLE001
            rep.fail(f"{path.name} invalid: {exc}")

    bundle = load_json(BUNDLE_PATH)
    try:
        Draft202012Validator.check_schema(bundle)
        rep.ok(f"{BUNDLE_PATH.name} is a valid schema (top-level keys: {sorted(bundle)})")
    except Exception as exc:  # noqa: BLE001
        rep.fail(f"{BUNDLE_PATH.name} invalid: {exc}")

    # Enums and team mapping in the per-file schemas
    pr_defs = schemas["predict_response"]["$defs"]
    rep.check(pr_defs["Category"]["enum"] == yaml_categories, "Category enum: schema == YAML (same order)", "Category enum differs between schema and YAML")
    rep.check(pr_defs["Team"]["enum"] == yaml_teams, "Team enum: schema == YAML (same order)", "Team enum differs between schema and YAML")
    req_defs = schemas["predict_request"]["$defs"]
    rep.check(req_defs["Channel"]["enum"] == yaml_channels, "Channel enum: schema == YAML", "Channel enum differs between schema and YAML")
    rules = schema_team_rules(pr_defs)
    rep.check(rules == table, "schema if/then team rules == YAML category->team table", f"team rules differ: schema={rules} yaml={table}")

    # The bundle must agree with the per-file schemas
    bundle_defs = bundle.get("$defs", {})
    if bundle_defs:
        for key, expected in (("Category", yaml_categories), ("Team", yaml_teams), ("Channel", yaml_channels)):
            if key in bundle_defs:
                rep.check(bundle_defs[key]["enum"] == expected, f"bundle {key} enum matches", f"bundle {key} enum differs")
        rep.ok(f"bundle $defs: {sorted(bundle_defs)}")
    else:
        rep.warn("bundle has no top-level $defs; skipped bundle enum comparison")
    return schemas, bundle


def validators(schemas: dict[str, dict]) -> dict[str, Draft202012Validator]:
    return {name: Draft202012Validator(s) for name, s in schemas.items()}


def collect_examples(spec: dict):
    """Yield (label, schema_name, instance) for every example in the OpenAPI YAML."""
    paths = spec["paths"]

    def content_examples(node):
        media = node.get("content", {}).get("application/json", {})
        if "example" in media:
            yield "example", media["example"]
        for key, ex in media.get("examples", {}).items():
            yield key, ex["value"]

    def resolve(ref_node):
        if "$ref" in ref_node:
            name = ref_node["$ref"].split("/")[-1]
            return spec["components"]["responses"][name]
        return ref_node

    p = paths["/predict"]["post"]
    for k, v in content_examples(p["requestBody"]):
        yield f"POST /predict request '{k}'", "predict_request", v
    for k, v in content_examples(p["responses"]["200"]):
        yield f"POST /predict 200 '{k}'", "predict_response", v

    b = paths["/predict/batch"]["post"]
    for k, v in content_examples(b["requestBody"]):
        yield f"POST /predict/batch request '{k}'", "batch_request", v
    for k, v in content_examples(b["responses"]["200"]):
        yield f"POST /predict/batch 200 '{k}'", "batch_response", v

    j = paths["/batch/jobs"]["post"]
    for k, v in content_examples(j["requestBody"]):
        yield f"POST /batch/jobs request '{k}'", "batch_job_request", v
    for k, v in content_examples(j["responses"]["202"]):
        yield f"POST /batch/jobs 202 '{k}'", "batch_job_status", v

    for k, v in content_examples(paths["/batch/jobs/{job_id}"]["get"]["responses"]["200"]):
        yield f"GET /batch/jobs/id 200 '{k}'", "batch_job_status", v
    for k, v in content_examples(paths["/batch/jobs/{job_id}/results"]["get"]["responses"]["200"]):
        yield f"GET /batch/jobs/id/results 200 '{k}'", "batch_job_results", v

    for k, v in content_examples(paths["/health"]["get"]["responses"]["200"]):
        yield f"GET /health 200 '{k}'", "health_response", v
    for k, v in content_examples(paths["/health"]["get"]["responses"]["503"]):
        yield f"GET /health 503 '{k}'", "health_response", v

    for name, resp in spec["components"]["responses"].items():
        for k, v in content_examples(resp):
            yield f"error response {name} '{k}'", "error_response", v


def verify_examples(rep: Report, spec: dict, vals: dict[str, Draft202012Validator]):
    section("OpenAPI examples vs JSON Schemas")
    n = 0
    for label, schema_name, instance in collect_examples(spec):
        errors = sorted(vals[schema_name].iter_errors(instance), key=lambda e: list(e.path))
        n += 1
        if errors:
            e = errors[0]
            rep.fail(f"{label} -> {schema_name}: {e.message} at {list(e.path)}")
        else:
            rep.ok(f"{label} -> {schema_name}")
    rep.ok(f"{n} examples checked")

    section("Negative controls (rules must reject bad responses)")
    good = {
        "category": "delivery_delay",
        "secondary_category": "payment_refund",
        "team": "Delivery Operations",
        "is_urgent": False,
        "confidence": 0.8,
        "model_version": "v0",
    }
    v = vals["predict_response"]
    cases = {
        "wrong team for category": {**good, "team": "Tech Support"},
        "secondary equals primary": {**good, "secondary_category": "delivery_delay"},
        "spam marked urgent": {**good, "category": "spam_irrelevant", "team": "Auto-close / Spam Filter", "secondary_category": None, "is_urgent": True},
        "spam with secondary": {**good, "category": "spam_irrelevant", "team": "Auto-close / Spam Filter", "is_urgent": False},
        "missing secondary key": {k: x for k, x in good.items() if k != "secondary_category"},
        "confidence above 1": {**good, "confidence": 1.2},
        "unknown category": {**good, "category": "billing"},
    }
    rep.check(v.is_valid(good), "baseline good response is accepted", "baseline good response was rejected")
    for label, bad in cases.items():
        rep.check(not v.is_valid(bad), f"rejected: {label}", f"NOT rejected: {label}")

    req = vals["predict_request"]
    req_cases = {
        "empty text": {"channel": "chat", "text": ""},
        "whitespace-only text": {"channel": "chat", "text": "  \n\t "},
        "unknown channel": {"channel": "sms", "text": "hi"},
        "missing channel": {"text": "hi"},
        "text not a string": {"channel": "chat", "text": 5},
        "text too long": {"channel": "chat", "text": "a" * 10001},
    }
    for label, bad in req_cases.items():
        rep.check(not req.is_valid(bad), f"request rejected: {label}", f"request NOT rejected: {label}")
    for label, ok_req in {
        "emoji-only text": {"channel": "chat", "text": "\U0001F64F\U0001F64F"},
        "sinhala text": {"channel": "chat", "text": "කාර් එකේ මගේ බෑග් එක අමතක වුණා"},
        "extra unknown property": {"channel": "email", "text": "hi", "language": "en"},
    }.items():
        rep.check(req.is_valid(ok_req), f"request accepted: {label}", f"request wrongly rejected: {label}")


# ------------------------------------------------------------------------ dataset


def read_csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = list(reader)
    return header, rows


def read_jsonl_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def norm_csv_row(r: dict) -> dict:
    return {
        "ticket_id": r["ticket_id"],
        "channel": r["channel"],
        "subject": r["subject"],
        "text": r["text"],
        "language": r["language"],
        "category": r["category"],
        "secondary_category": r["secondary_category"] or None,
        "is_urgent": {"true": True, "false": False}[r["is_urgent"]],
    }


def verify_dataset(rep: Report, table: dict[str, str], vals: dict[str, Draft202012Validator], categories, channels):
    section("Dataset (data/)")
    if not (DATA_DIR / "train.csv").is_file():
        rep.fail("data/train.csv not found. Download the dataset from the official link in README.md into data/")
        return

    notes = DATA_DIR / "DATA_NOTES.md"
    rep.check(notes.is_file(), "DATA_NOTES.md present", "DATA_NOTES.md missing")

    splits: dict[str, list[dict]] = {}
    for split, expected in EXPECTED_ROWS.items():
        csv_path = DATA_DIR / f"{split}.csv"
        jsonl_path = DATA_DIR / f"{split}.jsonl"
        header, csv_raw = read_csv_rows(csv_path)
        jl = read_jsonl_rows(jsonl_path)
        rep.check(header == DATA_COLUMNS, f"{split}.csv columns match the documented order", f"{split}.csv columns: {header}")
        rep.check(len(csv_raw) == expected, f"{split}.csv has {len(csv_raw)} rows", f"{split}.csv has {len(csv_raw)} rows, expected {expected}")
        rep.check(len(jl) == expected, f"{split}.jsonl has {len(jl)} rows", f"{split}.jsonl has {len(jl)} rows, expected {expected}")

        csv_rows = [norm_csv_row(r) for r in csv_raw]
        rep.check(csv_rows == jl, f"{split}: CSV and JSONL contents are identical (incl. embedded newlines)", f"{split}: CSV and JSONL differ")
        splits[split] = csv_rows

        nl = sum("\n" in r["text"] for r in csv_rows)
        rep.ok(f"{split}: {nl} texts contain embedded newlines (parsed correctly by the csv module)")
        non_nfc = sum(unicodedata.normalize("NFC", r["text"]) != r["text"] for r in csv_rows)
        rep.ok(f"{split}: {non_nfc} texts are not NFC-normalized (keep raw text; normalize deliberately in features)")

    all_rows = [(s, r) for s, rows in splits.items() for r in rows]

    # Label and field rules
    cat_set, chan_set = set(categories), set(channels)
    bad = Counter()
    for _, r in all_rows:
        if r["category"] not in cat_set:
            bad["category not in enum"] += 1
        if r["secondary_category"] is not None and r["secondary_category"] not in cat_set:
            bad["secondary not in enum"] += 1
        if r["secondary_category"] == r["category"]:
            bad["secondary == primary"] += 1
        if r["channel"] not in chan_set:
            bad["channel not in enum"] += 1
        if r["language"] not in LANGUAGES:
            bad["unknown language"] += 1
        if r["category"] == "spam_irrelevant" and (r["is_urgent"] or r["secondary_category"]):
            bad["spam urgent/secondary"] += 1
        if r["channel"] != "email" and r["subject"]:
            bad["non-email has subject"] += 1
        if not r["text"].strip():
            bad["blank text"] += 1
    rep.check(not bad, "all labels/fields are valid (enums, secondary != primary, spam rule, channel rule)", f"label problems: {dict(bad)}")

    # Uniqueness and leakage
    ids = [r["ticket_id"] for _, r in all_rows]
    rep.check(len(ids) == len(set(ids)), "ticket_ids unique across train+validation", "duplicate ticket_ids found")
    tr_texts = {r["text"] for r in splits["train"]}
    overlap = sum(r["text"] in tr_texts for r in splits["validation"])
    rep.check(overlap == 0, "no train/validation text overlap", f"{overlap} validation texts also appear in train")
    dup_in_train = len(splits["train"]) - len(tr_texts)
    rep.check(dup_in_train == 0, "no duplicate texts inside train", f"{dup_in_train} duplicate texts inside train")

    # Distribution summary (informational)
    for split, rows in splits.items():
        c = Counter(r["category"] for r in rows)
        sec = sum(r["secondary_category"] is not None for r in rows)
        urg = sum(r["is_urgent"] for r in rows)
        rep.ok(f"{split}: secondary on {sec / len(rows):.1%}, urgent {urg / len(rows):.1%}, languages {dict(Counter(r['language'] for r in rows))}")
        rep.ok(f"{split}: categories {dict(c.most_common())}")

    # Mapping: rows -> PredictRequest, labels -> PredictResponse
    req, resp = vals["predict_request"], vals["predict_response"]
    bad_req = bad_resp = 0
    for _, r in all_rows:
        request = {"ticket_id": r["ticket_id"], "channel": r["channel"], "subject": r["subject"], "text": r["text"]}
        if not req.is_valid(request):
            bad_req += 1
        gold = {
            "ticket_id": r["ticket_id"],
            "category": r["category"],
            "secondary_category": r["secondary_category"],
            "team": table[r["category"]],
            "is_urgent": r["is_urgent"],
            "confidence": 1.0,
            "model_version": "gold",
        }
        if not resp.is_valid(gold):
            bad_resp += 1
    rep.check(bad_req == 0, f"all {len(all_rows)} rows map to a valid PredictRequest (language column intentionally not sent)", f"{bad_req} rows fail the PredictRequest schema")
    rep.check(bad_resp == 0, f"all {len(all_rows)} gold labels map to a valid PredictResponse (team derived from category)", f"{bad_resp} gold labels fail the PredictResponse schema")


# --------------------------------------------------------------------------- main


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-data", action="store_true", help="skip dataset checks")
    args = ap.parse_args()

    rep = Report()
    spec, table, categories, teams, channels = verify_spec(rep)
    schemas, _bundle = verify_schemas(rep, spec, table, categories, teams, channels)
    vals = validators(schemas)
    verify_examples(rep, spec, vals)
    if not args.no_data:
        verify_dataset(rep, table, vals, categories, channels)

    print(f"\nSummary: {rep.failures} failure(s), {rep.warnings} warning(s)")
    return 1 if rep.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
