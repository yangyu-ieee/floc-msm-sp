"""Verify the R1.3 trained-residual aggregation reported in the manuscript.

The reported quantiles are computed per seed (128,000 coordinates per seed)
and then averaged across five seeds. They are not pooled quantiles over all
640,000 coordinates. This script checks the raw archived records, source-code
provenance, and the numerical strings in the canonical main manuscript.
"""
from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_CANDIDATES = (
    ROOT / "logs" / "r1_actual_residual_diagnostics_v2" / "results.json",
    ROOT / "results" / "diagnostics" / "actual_residuals" / "results.json",
)
CONFIG_CANDIDATES = (
    ROOT / "logs" / "r1_actual_residual_diagnostics_v2" / "config.json",
    ROOT / "results" / "diagnostics" / "actual_residuals" / "config.json",
)
MAIN_CANDIDATES = (
    ROOT / "manuscript" / "sp_main.tex",
    ROOT / "manuscript" / "sp_main.tex",
)
RESULTS = next((p for p in RESULT_CANDIDATES if p.exists()), RESULT_CANDIDATES[0])
CONFIG = next((p for p in CONFIG_CANDIDATES if p.exists()), CONFIG_CANDIDATES[0])
MAIN = next((p for p in MAIN_CANDIDATES if p.exists()), MAIN_CANDIDATES[0])
SCRIPT_CANDIDATES = (
    ROOT / "code" / "r1_actual_residual_diagnostics.py",
)
RUNNER = next((p for p in SCRIPT_CANDIDATES if p.exists()), SCRIPT_CANDIDATES[0])

LOSSES = ("mse", "huber", "floc_1.2", "charbonnier", "l1")
SEEDS = tuple(range(5))
COORDINATES_PER_SEED = 128_000
EXPECTED_TEXT = (
    "128,000 held-out coordinates per seed",
    "averaged over five seeds (not pooled)",
    "0.763 to 0.814",
    "1.031 to 1.047",
    "3.182\\% (MSE)",
    "2.522\\% (Huber)",
    "0.278\\% ($p$-moment loss, $p=1.2$)",
    "0.115\\% (Charbonnier)",
    "0.100\\% (L1)",
    "0.031, 0.014, 0.060, 0.061, and 0.068",
)


def mean_for(rows: list[dict], loss: str, group: str, field: str) -> float:
    values = [row[group][field] for row in rows if row["loss"] == loss]
    if len(values) != len(SEEDS):
        raise AssertionError(f"{loss}/{group}/{field}: expected 5 values, got {len(values)}")
    return float(statistics.mean(values))


def main() -> None:
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    rows = data["records"]
    if len(rows) != len(LOSSES) * len(SEEDS):
        raise AssertionError(f"Expected 25 records, got {len(rows)}")
    if config.get("seeds") != len(SEEDS):
        raise AssertionError(f"Unexpected seed count: {config.get('seeds')}")
    if hashlib.sha256(RUNNER.read_bytes()).hexdigest() != config["script_sha256"]:
        raise AssertionError("Runner source hash does not match archived config")

    for loss in LOSSES:
        loss_rows = [row for row in rows if row["loss"] == loss]
        if sorted(row["seed"] for row in loss_rows) != list(SEEDS):
            raise AssertionError(f"Unexpected seed IDs for {loss}")
        if any(row["n_residuals"] != COORDINATES_PER_SEED for row in loss_rows):
            raise AssertionError(f"Unexpected coordinate count for {loss}")

    q99 = {loss: mean_for(rows, loss, "residual", "abs_q99") for loss in LOSSES}
    q999 = {loss: mean_for(rows, loss, "residual", "abs_q999") for loss in LOSSES}
    score_share = {
        loss: 100.0 * mean_for(rows, loss, "score", "top_0p1pct_squared_score_share")
        for loss in LOSSES
    }
    grad_mean = {
        loss: mean_for(rows, loss, "heldout_parameter_gradient_norm", "mean")
        for loss in LOSSES
    }

    expected_ranges = (
        (min(q99.values()), max(q99.values()), 0.763, 0.814),
        (min(q999.values()), max(q999.values()), 1.031, 1.047),
    )
    for observed_min, observed_max, expected_min, expected_max in expected_ranges:
        if round(observed_min, 3) != expected_min or round(observed_max, 3) != expected_max:
            raise AssertionError(
                f"Seed-mean quantile range mismatch: {observed_min}, {observed_max}"
            )

    text = MAIN.read_text(encoding="utf-8")
    missing = [phrase for phrase in EXPECTED_TEXT if phrase not in text]
    if missing:
        raise AssertionError(f"Main text missing audited aggregation/value strings: {missing}")

    expected_score = {
        "mse": 3.182,
        "huber": 2.522,
        "floc_1.2": 0.278,
        "charbonnier": 0.115,
        "l1": 0.100,
    }
    expected_grad = {
        "mse": 0.031,
        "huber": 0.014,
        "floc_1.2": 0.060,
        "charbonnier": 0.061,
        "l1": 0.068,
    }
    for loss in LOSSES:
        if round(score_share[loss], 3) != expected_score[loss]:
            raise AssertionError(f"Score-share mismatch for {loss}: {score_share[loss]}")
        if round(grad_mean[loss], 3) != expected_grad[loss]:
            raise AssertionError(f"Gradient-mean mismatch for {loss}: {grad_mean[loss]}")

    output = {
        "scope": "R1.3 residual quantile, score-share, and total-gradient aggregation audit; no retraining.",
        "aggregation": "Compute each statistic per seed on 128000 coordinates, then average across five seeds; do not pool coordinates for quantiles.",
        "n_records": len(rows),
        "coordinates_per_seed": COORDINATES_PER_SEED,
        "q99_seed_mean_by_loss": q99,
        "q999_seed_mean_by_loss": q999,
        "across_loss_q99_range_rounded_3dp": [round(min(q99.values()), 3), round(max(q99.values()), 3)],
        "across_loss_q999_range_rounded_3dp": [round(min(q999.values()), 3), round(max(q999.values()), 3)],
        "top_0p1pct_squared_score_share_percent_seed_mean": score_share,
        "heldout_parameter_gradient_norm_seed_mean": grad_mean,
        "result_sha256": hashlib.sha256(RESULTS.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(RUNNER.read_bytes()).hexdigest(),
        "main_sha256": hashlib.sha256(MAIN.read_bytes()).hexdigest(),
    }
    out = RESULTS.with_name("residual_quantile_aggregation_audit.json")
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("PASS: 25 seed/loss records; per-seed quantiles are averaged, not pooled")
    print(f"q99 range={output['across_loss_q99_range_rounded_3dp']}; q999 range={output['across_loss_q999_range_rounded_3dp']}")
    print(f"result SHA-256={output['result_sha256']}")
    print(f"summary={out}")


if __name__ == "__main__":
    main()
