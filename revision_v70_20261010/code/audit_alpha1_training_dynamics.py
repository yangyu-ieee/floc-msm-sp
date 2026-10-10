"""Verify archived alpha=1 trajectories against Supplementary Table S16.

This is a deterministic transcription/provenance audit; it does not retrain.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "alpha1_training_dynamics"
SUPPLEMENT = ROOT / "paper_tables" / "diagnostic_table_snapshots.tex"
LOSSES = ("mse", "l1", "charbonnier", "floc_1.2")
LABELS = {"mse": "MSE", "l1": "L1", "charbonnier": "Charbonnier", "floc_1.2": r"$p$-moment loss ($p=1.2$)"}


def near(actual: float, expected: float, digits: int) -> bool:
    return round(actual, digits) == round(expected, digits)


def main() -> None:
    config = json.loads((RUN / "config.json").read_text(encoding="utf-8"))
    trajectories_path = RUN / "trajectories.json"
    trajectories = json.loads(trajectories_path.read_text(encoding="utf-8"))
    summary = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
    seeds, epochs = int(config["seeds"]), int(config["epochs"])
    expected_keys = {f"seed={seed}/loss={loss}" for seed in range(seeds) for loss in LOSSES}
    if set(trajectories) != expected_keys or len(expected_keys) != 20:
        raise AssertionError("Expected exactly 20 seed/loss trajectories")
    if hashlib.sha256(trajectories_path.read_bytes()).hexdigest() != summary["trajectory_sha256"]:
        raise AssertionError("Trajectory SHA-256 does not match archived summary")
    runner = ROOT / "code" / "alpha1_training_dynamics" / "r3_2_alpha1_training_dynamics.py"
    base_runner = ROOT / "code" / "alpha1_training_dynamics" / "targeted_alpha10_fullseed.py"
    if hashlib.sha256(runner.read_bytes()).hexdigest() != config["runner_sha256"]:
        raise AssertionError("Packaged S16 runner does not match recorded source hash")
    public_base_hash = config.get("public_base_runner_sha256")
    if not public_base_hash:
        raise AssertionError("Public portability-patched S16 base-runner hash is not recorded")
    if hashlib.sha256(base_runner.read_bytes()).hexdigest() != public_base_hash:
        raise AssertionError("Public S16 base runner does not match its recorded portability-copy hash")

    text = SUPPLEMENT.read_text(encoding="utf-8")
    start = text.find(r"\label{tab:alpha1_training_dynamics}")
    end = text.find(r"\end{tabular}", start)
    if start < 0 or end < 0:
        raise AssertionError("Supplementary Table S16 not found")
    table = text[start:end]
    checked = 0
    for loss in LOSSES:
        runs = [trajectories[f"seed={seed}/loss={loss}"] for seed in range(seeds)]
        if any(len(rows) != epochs + 1 for rows in runs):
            raise AssertionError(f"Incomplete epoch trajectory for {loss}")
        if any([row["epoch"] for row in rows] != list(range(epochs + 1)) for rows in runs):
            raise AssertionError(f"Unexpected epoch labels for {loss}")
        reductions = [100 * (1 - rows[-1]["heldout_fixed_mask_loss"] / rows[0]["heldout_fixed_mask_loss"]) for rows in runs]
        q1 = [rows[1]["grad_norm_q90"] for rows in runs]
        q20 = [rows[-1]["grad_norm_q90"] for rows in runs]
        if any(rows[-1]["heldout_fixed_mask_loss"] >= rows[0]["heldout_fixed_mask_loss"] for rows in runs):
            raise AssertionError(f"Held-out objective did not fall in every {loss} trajectory")
        if any(rows[-1]["grad_norm_q90"] >= rows[1]["grad_norm_q90"] for rows in runs):
            raise AssertionError(f"Gradient q90 did not fall in every {loss} trajectory")
        if any(row["nonfinite_steps"] for rows in runs for row in rows):
            raise AssertionError(f"Non-finite update recorded for {loss}")

        line = next((line.strip().rstrip("\\") for line in table.splitlines() if line.strip().startswith(LABELS[loss] + " &")), None)
        if line is None:
            raise AssertionError(f"Table row missing for {loss}")
        cells = [part.strip() for part in line.split("&")]
        if len(cells) != 4:
            raise AssertionError(f"Unexpected S16 row format: {line}")
        vals = [[float(x) for x in re.findall(r"\d+\.\d+", cell)] for cell in cells[1:]]
        expected = [
            [round(sorted(reductions)[2], 1), round(min(reductions), 1), round(max(reductions), 1)],
            [round(sorted(q1)[2], 5), round(min(q1), 5), round(max(q1), 5)],
            [round(sorted(q20)[2], 5), round(min(q20), 5), round(max(q20), 5)],
        ]
        if any(len(got) != 3 or any(not near(a, b, 1 if i == 0 else 5) for a, b in zip(got, want)) for i, (got, want) in enumerate(zip(vals, expected))):
            raise AssertionError(f"S16 row mismatch for {loss}: {vals} != {expected}")
        checked += 1

    print(f"PASS: {checked} S16 rows match all {len(expected_keys)} archived trajectories")
    print(f"PASS: trajectory SHA-256 {summary['trajectory_sha256']}")
    print(f"Original run-source SHA-256 retained in config: {config['base_runner_sha256']}")
    print("Note: public base-runner copy changes only the machine-specific default output path.")
    print("Scope: archived-run integrity and table transcription; no retraining or convergence claim")


if __name__ == "__main__":
    main()
