"""Check and optionally refresh the main-table summary from archived results.

This script does not train models. It verifies that the summary contains the
same per-seed records as results.json and restores the omitted scratch entry.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "main_table_reconciliation_20261009"
RESULTS = RUN / "results.json"
SUMMARY = RUN / "summary.json"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="copy the archived scratch result into summary.json")
    args = parser.parse_args()

    results = load(RESULTS)
    summary = load(SUMMARY)
    if "scratch" not in results:
        raise SystemExit("results.json has no scratch baseline")
    for key, value in summary.items():
        if key not in results or value != results[key]:
            raise SystemExit(f"summary entry does not match results.json: {key}")
    if "scratch" in summary and summary["scratch"] != results["scratch"]:
        raise SystemExit("scratch summary does not match results.json")
    if args.write and "scratch" not in summary:
        summary["scratch"] = results["scratch"]
        SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    elif args.write:
        SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    elif "scratch" not in summary:
        raise SystemExit("summary.json omits scratch; rerun with --write to synchronize")
    print(f"PASS: {len(summary)} summary groups match archived results, including scratch")


if __name__ == "__main__":
    main()
