"""Audit the finite-sample residual-quantile statement in the manuscript.

This checks archived stagewise snapshots and source wording. It does not
retrain models or infer a population residual law or network-gradient order.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "diagnostics" / "stagewise" / "results.json"
MAIN_SOURCE = ROOT / "manuscript" / "sp_main.tex"
SUPPLEMENT_SOURCE = ROOT / "supplement" / "sp_supplementary.tex"
EXPECTED_RECORDS = 100  # 5 losses x 5 seeds x 4 epochs
EXPECTED_RESIDUALS = 128_000
P = 1.2
R_STAR = (P / 2.0) ** (1.0 / (2.0 - P))
EXPECTED_MIN = 0.30298790037631995
EXPECTED_MAX = 0.4261419057846073


def main() -> None:
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    records = data["records"]
    if len(records) != EXPECTED_RECORDS:
        raise AssertionError(f"Expected {EXPECTED_RECORDS} records; got {len(records)}")
    keys = {(r["loss"], r["seed"], r["epoch"]) for r in records}
    if len(keys) != EXPECTED_RECORDS:
        raise AssertionError("Duplicate stagewise loss/seed/epoch records")

    q90_values: list[float] = []
    for record in records:
        if int(record["n_residuals"]) != EXPECTED_RESIDUALS:
            raise AssertionError("Unexpected residual-coordinate count")
        value = float(record["residual"]["abs_q90"])
        if not math.isfinite(value) or value < 0:
            raise AssertionError("Non-finite or negative absolute-residual q90")
        if not value < R_STAR:
            raise AssertionError(f"q90={value} is not below crossover r*={R_STAR}")
        q90_values.append(value)

    observed_min, observed_max = min(q90_values), max(q90_values)
    if not math.isclose(observed_min, EXPECTED_MIN, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError(f"Minimum q90 changed: {observed_min}")
    if not math.isclose(observed_max, EXPECTED_MAX, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError(f"Maximum q90 changed: {observed_max}")

    main_text = MAIN_SOURCE.read_text(encoding="utf-8")
    supplement_text = SUPPLEMENT_SOURCE.read_text(encoding="utf-8")
    claim = "0.3030 to 0.4261"
    for label, source in (("main", main_text), ("Supplement", supplement_text)):
        if claim not in source:
            raise AssertionError(f"Reported q90 range missing from {label} source")

    output = {
        "audit_scope": "Archived finite-sample residual quantiles only; no population-law or full-gradient inference.",
        "result_sha256": hashlib.sha256(RESULTS.read_bytes()).hexdigest(),
        "main_source_sha256": hashlib.sha256(MAIN_SOURCE.read_bytes()).hexdigest(),
        "supplement_source_sha256": hashlib.sha256(SUPPLEMENT_SOURCE.read_bytes()).hexdigest(),
        "record_count": len(records),
        "residual_coordinates_per_record": EXPECTED_RESIDUALS,
        "p": P,
        "score_crossover": R_STAR,
        "absolute_residual_q90_min": observed_min,
        "absolute_residual_q90_max": observed_max,
        "records_below_crossover": sum(v < R_STAR for v in q90_values),
        "source_range_present_in_main_and_supplement": True,
    }
    output_path = RESULTS.with_name("score_crossover_snapshot_audit.json")
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        f"PASS: {len(records)} snapshots; q90 range {observed_min:.10f}--"
        f"{observed_max:.10f}, all below r*={R_STAR:.10f}; source wording matches"
    )
    print(f"Summary: {output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
