"""Recompute and verify Supplementary Table S15 from archived stagewise logs.

This is a deterministic transcription audit, not a retraining run or an
independent assessment of optimizer stability.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_CANDIDATES = (
    ROOT / "logs" / "r1_stagewise_diagnostics_v2" / "results.json",
    ROOT / "results" / "diagnostics" / "stagewise" / "results.json",
)
RESULT_PATH = next((path for path in RESULT_CANDIDATES if path.exists()), RESULT_CANDIDATES[0])
TABLE_SNAPSHOT = ROOT / "paper_tables" / "diagnostic_table_snapshots.tex"
SEEDS = 5
EPOCHS = (1, 5, 10, 20)
LOSSES = ("mse", "huber", "floc_1.2", "charbonnier", "l1")
TAILS = ("q50", "q90", "max")
DISPLAY_RE = re.compile(r"^(\d+) & (MSE|Huber|\$p\$-moment loss \(\$p=1\.2\$\)|Charbonnier|L1) & (.+) & (.+) & (.+)$")


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def population_sd(values: list[float]) -> float:
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def main() -> None:
    data = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    records = data["records"]
    expected_rows = len(EPOCHS) * len(LOSSES) * SEEDS
    if len(records) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} stagewise records; got {len(records)}")
    keys = [(r["epoch"], r["loss"], r["seed"]) for r in records]
    if len(set(keys)) != expected_rows:
        raise AssertionError("Duplicate or missing epoch/loss/seed record")

    summary: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for epoch in EPOCHS:
        summary[str(epoch)] = {}
        for loss in LOSSES:
            seed_records = [r for r in records if r["epoch"] == epoch and r["loss"] == loss]
            seed_records.sort(key=lambda r: r["seed"])
            if len(seed_records) != SEEDS or [r["seed"] for r in seed_records] != list(range(SEEDS)):
                raise AssertionError(f"Unexpected seed records for epoch={epoch}, loss={loss}")
            summary[str(epoch)][loss] = {}
            for tail in TAILS:
                values = []
                for record in seed_records:
                    gradient = record["parameter_gradient_norm"]
                    batch_values = gradient["batch_values"]
                    if gradient["n_batches"] != 8 or len(batch_values) != 8:
                        raise AssertionError("Expected eight held-out batch-gradient norms per seed")
                    if not all(math.isfinite(float(x)) and float(x) >= 0 for x in batch_values):
                        raise AssertionError("Non-finite/negative batch-gradient norm")
                    values.append(float(gradient[tail]))
                summary[str(epoch)][loss][tail] = {
                    "mean": mean(values),
                    "sd_ddof0": population_sd(values),
                    "seed_values": values,
                }

    tex = TABLE_SNAPSHOT.read_text(encoding="utf-8")
    if r"\label{tab:stagewise_gradient_tails}" not in tex:
        raise AssertionError("Supplementary Table S15 label is missing")
    table_start = tex.index(r"\label{tab:stagewise_gradient_tails}")
    table_end = tex.index(r"\end{table}", table_start)
    table = tex[table_start:table_end]
    displayed = 0
    loss_labels = {
        "MSE": "mse", "Huber": "huber", "$p$-moment loss ($p=1.2$)": "floc_1.2",
        "Charbonnier": "charbonnier", "L1": "l1",
    }
    for line in table.splitlines():
        normalized = re.sub(r"\\+$", "", line.strip()).rstrip()
        match = DISPLAY_RE.match(normalized)
        if not match:
            continue
        epoch, label, *cells = match.groups()
        epoch, loss = str(int(epoch)), loss_labels[label]
        if len(cells) != len(TAILS):
            raise AssertionError("Unexpected displayed gradient-tail columns")
        for tail, cell in zip(TAILS, cells, strict=True):
            nums = [float(x) for x in re.findall(r"\d+\.\d+", cell)]
            expected = summary[epoch][loss][tail]
            if len(nums) != 2:
                raise AssertionError(f"Could not parse displayed cell: {cell}")
            if round(expected["mean"], 4) != round(nums[0], 4) or round(expected["sd_ddof0"], 4) != round(nums[1], 4):
                raise AssertionError(f"Table mismatch at epoch={epoch}, loss={loss}, tail={tail}: {nums} != {expected}")
            displayed += 1
    if displayed != len(EPOCHS) * len(LOSSES) * len(TAILS):
        raise AssertionError(f"Expected 60 checked displayed cells; got {displayed}")

    output = {
        "audit_scope": "Table transcription and provenance only; no retraining and no independent optimizer-stability claim.",
        "result_sha256": hashlib.sha256(RESULT_PATH.read_bytes()).hexdigest(),
        "source_provenance_sha256": hashlib.sha256((ROOT / "SOURCE_PROVENANCE.json").read_bytes()).hexdigest(),
        "table_snapshot_sha256": hashlib.sha256(TABLE_SNAPSHOT.read_bytes()).hexdigest(),
        "record_count": len(records),
        "seed_count": SEEDS,
        "batches_per_seed_snapshot": 8,
        "displayed_cells_checked": displayed,
        "summary": summary,
    }
    out_path = RESULT_PATH.with_name("stagewise_gradient_tail_summary.json")
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"PASS: {len(records)} records; {displayed} displayed Table S15 cells match archived results")
    print(f"Summary: {out_path.relative_to(ROOT)}")
    print("Scope: deterministic transcription/provenance audit only; not retraining or an independent claim review")


if __name__ == "__main__":
    main()
